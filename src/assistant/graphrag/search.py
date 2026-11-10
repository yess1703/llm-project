"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json
from itertools import combinations

import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase

REFUSAL = "в базе знаний не найдено"


def _documents(sources) -> list[str]:
    """Чанк идентифицирует свидетельство, наружу возвращается документ."""
    return sorted({source.split("#", 1)[0] for source in sources})


def _replacement_edges(graph: nx.DiGraph, nodes) -> set[tuple[str, str]]:
    """Все продолжения замен, с защитой от циклов и ветвлений."""
    pending = [node for node in nodes if graph.nodes[node]["type"] == "Part"]
    visited = set()
    edges = set()
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        for target, data in graph[node].items():
            if data["type"] == "REPLACED_BY":
                edges.add((node, target))
                pending.append(target)
    return edges


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    graph = kb.graph
    codes = [f"{code_type(code)}:{code}" for code in find_codes(question)]
    if any(node not in graph for node in codes):
        return Answer(REFUSAL, refused=True)
    names = [f"{kind}:{name}" for kind, name in find_names(question, kb.vocabulary)]
    anchors = sorted(set(codes) | {node for node in names if node in graph})
    undirected = graph.to_undirected(as_view=True)

    def fact_groups():
        if len(anchors) >= 2:
            for source, target in combinations(anchors, 2):
                for path in nx.all_simple_paths(undirected, source, target, cutoff=3):
                    edges = set()
                    for a, b in zip(path, path[1:], strict=False):
                        # Поиск пути игнорирует направление, факт его сохраняет.
                        edges.update((u, v) for u, v in ((a, b), (b, a)) if graph.has_edge(u, v))
                    yield edges | _replacement_edges(graph, path)
        elif anchors:
            nodes = nx.single_source_shortest_path_length(undirected, anchors[0], cutoff=2)
            for source, target in sorted(graph.subgraph(nodes).edges):
                yield {(source, target)} | _replacement_edges(graph, (source, target))

    selected = set()
    facts = []
    used = 0
    for group in fact_groups():
        additions = sorted(group - selected)
        lines = [f"{graph.nodes[a]['name']} -{graph[a][b]['type']}-> {graph.nodes[b]['name']}" for a, b in additions]
        # Лимит включает номера, пробелы и переводы строк. Группа добавляется целиком.
        cost = sum(len(f"{len(facts) + i}. {line}\n") for i, line in enumerate(lines, start=1))
        if used + cost <= 8000:
            selected.update(additions)
            facts.extend(lines)
            used += cost
    if not facts:
        return Answer(REFUSAL, refused=True)

    replacements = [(a, b) for a, b in selected if graph[a][b]["type"] == "REPLACED_BY"]
    old_parts = {a for a, _ in replacements}
    latest = sorted({graph.nodes[b]["name"] for _, b in replacements if b not in old_parts})
    context = "\n".join(f"{index}. {fact}" for index, fact in enumerate(facts, start=1))
    reply = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Ты ассистент сервисной службы. Отвечай по-русски только по пронумерованным фактам. "
                    "Сохраняй коды и направление отношений. REPLACED_BY означает: старая деталь заменена новой. "
                    "При наличии замены назови последнюю деталь всей цепочки, не советуй заказывать старую. "
                    "При цикле или неоднозначной замене сообщи об этом. Не придумывай цены и остатки. "
                    f"Если данных для ответа нет, напиши: «{REFUSAL}»."
                ),
            },
            {"role": "user", "content": f"Факты:\n{context}\n\nВопрос: {question}"},
        ]
    )
    text = reply.content.strip()
    refused = not text or "не найдено" in text.lower()
    if not refused and latest and not set(latest).issubset(find_codes(text)):
        text += "\nПоследние детали в найденных цепочках замены: " + ", ".join(latest) + "."
    sources = _documents(source for a, b in selected for source in graph[a][b].get("sources", []))
    return Answer(text or REFUSAL, sources=sources, facts=facts, refused=refused)


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    partials = []
    sources = set()
    for community in kb.communities:
        if not community.summary.strip():
            continue
        reply = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Ответь на вопрос только по данной сводке сообщества. Верни JSON: answer — частичный ответ, "
                        "score — целое число от 0 до 10, полезность сводки для вопроса. "
                        "Если ответа в сводке нет, score должен быть 0. Не добавляй сведений извне."
                    ),
                },
                {"role": "user", "content": f"Сводка:\n{community.summary}\n\nВопрос: {question}"},
            ],
            json_schema=MAP_SCHEMA,
        )
        try:
            partial = json.loads(reply.content)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(partial, dict):
            continue
        answer, score = partial.get("answer"), partial.get("score")
        if not isinstance(answer, str) or not answer.strip() or type(score) is not int or not 0 < score <= 10:
            continue
        partials.append({"community_id": community.community_id, "answer": answer, "score": score})
        # У Community нет поля sources: восстанавливаем их из исходного подграфа.
        subgraph = kb.graph.subgraph(community.nodes)
        sources.update(source for _, data in subgraph.nodes(data=True) for source in data.get("sources", []))
        sources.update(source for _, _, data in subgraph.edges(data=True) for source in data.get("sources", []))

    if not partials:
        return Answer(REFUSAL, refused=True)
    reply = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Объедини частичные ответы на вопрос в один связный ответ по-русски. "
                    "Используй только переданные ответы; не придумывай факты и числа. "
                    "score — полезность, а не число отказов. Не суммируй показатели, которые могут пересекаться. "
                    f"Если данных недостаточно, напиши: «{REFUSAL}»."
                ),
            },
            {
                "role": "user",
                "content": (f"Частичные ответы:\n{json.dumps(partials, ensure_ascii=False)}\n\nВопрос: {question}"),
            },
        ]
    )
    text = reply.content.strip()
    return Answer(text or REFUSAL, sources=_documents(sources), refused=not text or "не найдено" in text.lower())
