import pandas as pd
import networkx as nx

from independence_tests.chi2_test import p_chi2_score
from utilities.graph_utilities import cap_parents, eliminate_directed_circles


class CDBase:
    """
    A base class for Causal Discovery algorithms.
    """
    def __init__(self, data: pd.DataFrame):
        """
        A base class for Causal Discovery algorithms.
        :param data: The dataset based on which the algorithm will perform causal discovery.
        """
        self.data: pd.DataFrame = data
        self.graph: nx.DiGraph = nx.DiGraph()

    def fit(self, **kwargs):
        raise NotImplementedError("The fit method must be implemented!")

    def _p_chi2_score_local(self, x_col: str, y_col: str, z_col: (list or tuple or str) = None, method: str = 'min'):
        return p_chi2_score(data=self.data, x_col=x_col, y_col=y_col, z_col=z_col, method=method)

    def cap_parents(self, max_parent_num: int, scoring_method: callable = None):
        """
            Caps the number of parents for every node in the graph by removing the edges with the HIGHEST scores.
            :param max_parent_num: The maximum number of parents for every node.
            :param scoring_method: A scoring method which takes the names of two nodes and returns an independence score.
            For example: score(node_name_1, node_name_2) -> chance_for_independence
            The independence score is higher if the two nodes are more likely to be independent.
            :return: A DiGraph where none of the nodes have more parents than the specified amount.
            """
        if scoring_method is None:
            scoring_method = self._p_chi2_score_local
        if self.graph is None:
            raise AttributeError("You must have a graph assigned to the CDBase object!")
        return cap_parents(graph=self.graph, max_parent_num=max_parent_num, scoring_method=scoring_method)

    def eliminate_directed_circles(self, scoring_method: callable = None):
        """
            Eliminates all directed circles in the graph by removing the edge which has the HIGHEST score from every directed circle.
            :param scoring_method: A scoring method which takes the names of two nodes and returns an independence score.
            For example: score(node_name_1, node_name_2) -> chance_for_independence
            The independence score is higher if the two nodes are more likely to be independent.
            :return: A DAG in form of a DiGraph.
            """
        if scoring_method is None:
            scoring_method = self._p_chi2_score_local
        if self.graph is None:
            raise AttributeError("You must have a graph assigned to the CDBase object!")
        return eliminate_directed_circles(graph=self.graph, scoring_method=scoring_method)

    def save_graph(self, path: str, parent_cap: int = None, eliminate_circles: bool = True, graph: nx.Graph = None):
        if graph is None:
            graph = self.graph.copy()
        if parent_cap is not None:
            graph = cap_parents(graph=graph, scoring_method=self._p_chi2_score_local, max_parent_num=parent_cap)
        if eliminate_circles:
            graph = eliminate_directed_circles(graph, scoring_method=self._p_chi2_score_local)
        nx.write_graphml_xml(graph, path)
        # print(graph, path)
