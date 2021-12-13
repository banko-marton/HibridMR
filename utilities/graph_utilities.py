import networkx as nx
import numpy as np


def cap_parents(graph: nx.DiGraph, max_parent_num: int, scoring_method: callable):
    """
    Caps the number of parents for every node in the graph by removing the edges with the HIGHEST scores.
    :param graph: The initial graph.
    :param max_parent_num: The maximum number of parents for every node.
    :param scoring_method: A scoring method which takes the names of two nodes and returns an independence score.
    For example: score(node_name_1, node_name_2) -> chance_for_independence
    The independence score is higher if the two nodes are more likely to be independent.
    :return: A DiGraph where none of the nodes have more parents than the specified amount.
    """
    result = graph.copy()
    for node in result.nodes:
        parents = list(result.predecessors(node))
        if len(parents) > max_parent_num:
            parent_scores = []
            for parent in parents:
                parent_scores.append(scoring_method(node, parent))
            sorted_indices = np.argsort(parent_scores)
            for i in range(max_parent_num, len(sorted_indices)):
                result.remove_edge(parents[sorted_indices[i]], node)
    return result


def eliminate_directed_circles(graph: nx.DiGraph, scoring_method: callable):
    """
    Eliminates all directed circles in the graph by removing the edge which has the HIGHEST score from every directed circle.
    :param graph: The initial graph.
    :param scoring_method: A scoring method which takes the names of two nodes and returns an independence score.
    For example: score(node_name_1, node_name_2) -> chance_for_independence
    The independence score is higher if the two nodes are more likely to be independent.
    :return: A DAG in form of a DiGraph.
    """
    result = graph.copy()
    while not nx.is_directed_acyclic_graph(result):
        cycle = nx.find_cycle(result)
        highest_score = np.NINF
        selected_edge = None
        for edge in cycle:
            edge_score = scoring_method(edge[0], edge[1])
            if edge_score > highest_score:
                highest_score = edge_score
                selected_edge = (edge[0], edge[1])
        result.remove_edge(selected_edge[0], selected_edge[1])
    return result


def find_paths(graph: nx.Graph, u, n):
    """
    Finds all the n-long non-circular paths from node u.
    :param graph: The input graph in which you want to find the paths.
    :param u: The starting node for the paths.
    :param n: The required length of the paths.
    :return: The list of all n-long non-circular paths from node u.
    """
    if n == 0:
        return [[u]]
    paths = [[u]+path for neighbor in graph.neighbors(u) for path in find_paths(graph, neighbor, n-1) if u not in path]
    return paths
