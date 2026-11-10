"""Сообщества графа и их сводки. Требование FR-3."""

import json
from collections import Counter

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    undirected = nx.Graph()
    undirected.add_nodes_from(sorted(graph.nodes))
    for source, target, data in sorted(graph.edges(data=True)):
        previous = undirected.get_edge_data(source, target, {}).get("weight", 0)
        undirected.add_edge(source, target, weight=previous + data.get("weight", 1))

    isolated = sorted(nx.isolates(undirected))
    connected = undirected.subgraph(set(undirected) - set(isolated))
    groups = [sorted(group) for group in nx.community.louvain_communities(connected, seed=7)] if connected else []
    groups.extend([node] for node in isolated)
    return [Community(index, nodes) for index, nodes in enumerate(sorted(groups))]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    subgraph = graph.subgraph(community.nodes)
    relation_weights = Counter()
    failures = Counter()
    facts = []
    for source, target, data in sorted(subgraph.edges(data=True)):
        kind, weight = data["type"], data.get("weight", 1)
        relation_weights[kind] += weight
        if kind == "FAILED":
            failures[subgraph.nodes[target]["name"]] += weight
        facts.append(
            {
                "fact": f"{subgraph.nodes[source]['name']} -{kind}-> {subgraph.nodes[target]['name']}",
                "weight": weight,
                "sources": sorted({item.split("#", 1)[0] for item in data.get("sources", [])}),
            }
        )
    context = {
        "entities": sorted(subgraph.nodes),
        "node_count": subgraph.number_of_nodes(),
        "edge_count": subgraph.number_of_edges(),
        "relation_weights": dict(sorted(relation_weights.items())),
        "failures_by_component": dict(sorted(failures.items())),
        "facts": facts,
    }
    reply = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Напиши по-русски сводку сообщества графа: сущности, связи, частые отказы и замены. "
                    "Не более 150 слов. Используй только переданные данные. Все числа уже рассчитаны кодом: "
                    "weight — число чанков со связью, failures_by_component — сумма весов FAILED по узлу. "
                    "Не считай связи вручную, не придумывай числа и не обобщай на весь корпус. "
                    "Сохрани коды и укажи документы-источники для основных выводов."
                ),
            },
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ]
    )
    return " ".join(reply.content.split()[:150])
