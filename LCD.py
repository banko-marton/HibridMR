import networkx as nx
import pandas as pd
import numpy as np
from itertools import combinations, chain
from tqdm import tqdm
import os

from cd_base import CDBase
from data.generate_test_data import StrictPartitionedRandomEdgeGenerator, RandomPartitionedValueCountGenerator
from utilities.graph_utilities import find_paths
from independence_tests.chi2_test import is_independent, p_chi2_score
from performance_testing.dag_test import test_method


class LCD(CDBase):
    def __init__(self, data: pd.DataFrame, save_folder: str = None):
        CDBase.__init__(self, data=data)
        self.skeleton: nx.DiGraph = nx.DiGraph()
        self.alpha = 0.05
        self.bon_correction = False
        self.v_scores = []
        self.parent_cap: int = None
        self.verbose = 0
        self.save_folder = save_folder

    def fit(self, alpha: float = 0.05, bonferroni_correction: bool = False, parent_cap: int = None, verbose: int = 0):
        """
        Attempts to find the original causal graph based on local estimates.
        :param alpha: The significance threshold for p-values.
        :param bonferroni_correction: Whether or not use Bonferroni correction.
        :param parent_cap: Sets the maximum number of parents every node could have.
        :param verbose: Sets the verbosity level. (0 = silent)
        :return: The reconstructed causal graph.
        """
        if self.save_folder is not None and not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)

        self.bon_correction = bonferroni_correction
        self.alpha = alpha
        self.verbose = verbose
        self.parent_cap = parent_cap

        self._detect_skeleton()
        self._calculate_v_scores()
        self._detect_v_structures()
        self._do_until_isomorphism(self._find_chains)
        self._find_common_parents()


        if self.parent_cap is not None:
            self.graph = self.cap_parents(max_parent_num=self.parent_cap)

        self.graph = self.eliminate_directed_circles()

        return self.graph

    def _do_until_isomorphism(self, method: callable):
        last_graph = self.graph.copy()
        method()
        current_graph = self.graph.copy()
        while not nx.is_isomorphic(last_graph, current_graph):
            last_graph = current_graph
            method()
            current_graph = self.graph.copy()

    def _detect_skeleton(self):
        self.skeleton = nx.Graph()
        self.skeleton.add_nodes_from(self.data.columns)
        all_variable_pairs = list(combinations(self.data.columns, 2))
        if self.bon_correction:
            alpha_local = self.alpha / len(all_variable_pairs)
        else:
            alpha_local = self.alpha
        for pair in all_variable_pairs:
            if not is_independent(data=self.data, x_col=pair[0], y_col=pair[1], alpha=alpha_local, method='min'):
                self.skeleton.add_edge(pair[0], pair[1])

        if self.save_folder is not None:
            self.save_graph(path=self.save_folder + "/lcd_skeleton.xml",
                            eliminate_circles=False,
                            graph=self.skeleton)

    @staticmethod
    def _find_all_paths(graph: nx.Graph):
        paths = set()
        for node in graph.nodes:
            paths_from_node = find_paths(graph, node, 2)
            paths_from_node = [tuple(sorted(element)) for element in paths_from_node]
            paths_from_node = set(paths_from_node)
            paths.update(paths_from_node)
        paths = [list(element) for element in paths]
        paths = np.array(paths)
        return paths

    def _calculate_v_scores(self):
        paths = self._find_all_paths(self.skeleton)

        v_scores_local = []
        if self.verbose:
            print("Calculating v-scores...")
            pbar = tqdm(total=len(paths))
        for path in paths:
            direct_scores = []
            indirect_scores = []
            for i in range(3):
                z = str(path[i])
                x, y = set(path) - {z}
                x, y = str(x), str(y)
                if not (self.skeleton.has_edge(z, x) and self.skeleton.has_edge(z, y)):
                    direct_scores.append(np.NINF)
                    indirect_scores.append(np.NINF)
                    continue
                p_direct = p_chi2_score(data=self.data, x_col=x, y_col=y)
                direct_scores.append(p_direct)
                p_indirect = p_chi2_score(data=self.data, x_col=x, y_col=y, z_col=z)
                indirect_scores.append(p_indirect)
            path_scores = list(chain(path, direct_scores, indirect_scores))
            v_scores_local.append(path_scores)
            if self.verbose:
                pbar.update(1)

        self.v_scores = v_scores_local

    def _detect_v_structures(self):
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.skeleton.nodes)

        if len(self.v_scores) == 0:
            return

        if self.bon_correction:
            alpha_local = self.alpha / float(len(self.v_scores))
        else:
            alpha_local = self.alpha

        nx.set_edge_attributes(self.skeleton, False, "conflict")

        for row in self.v_scores:
            path = row[0:3]
            direct_scores = row[3:6]
            indirect_scores = row[6:9]
            for i in range(3):
                if np.isinf(direct_scores[i]) or np.isinf(indirect_scores[i]):
                    continue
                if direct_scores[i] > alpha_local and indirect_scores[i] < alpha_local:
                    z = path[i]
                    x, y = set(path) - {z}
                    for neighbor in [x, y]:
                        if self.graph.has_edge(neighbor, z):
                            continue
                        if self.graph.has_edge(z, neighbor):
                            nx.set_edge_attributes(self.skeleton, {(z, neighbor): {"conflict": True}})
                        elif not self.skeleton[neighbor][z]["conflict"]:
                            self.graph.add_edge(neighbor, z)

        if self.save_folder is not None:
            self.save_graph(path=self.save_folder + "/01_v-structures.xml",
                            parent_cap=self.parent_cap,
                            eliminate_circles=True)

    def _find_chains(self):
        if len(self.v_scores) == 0:
            return

        if self.bon_correction:
            alpha_local = self.alpha / float(len(self.v_scores))
        else:
            alpha_local = self.alpha

        for row in self.v_scores:
            path = row[0:3]
            direct_scores = row[3:6]
            indirect_scores = row[6:9]
            for i in range(3):
                if np.isinf(direct_scores[i]) or np.isinf(indirect_scores[i]):
                    continue
                if direct_scores[i] < alpha_local and indirect_scores[i] > alpha_local:
                    z = path[i]
                    x, y = set(path) - {z}
                    added_edge = None
                    if self.graph.has_edge(x, z) and not (self.graph.has_edge(y, z) or self.graph.has_edge(z, y)):
                        self.graph.add_edge(z, y)
                        added_edge = (z, y)
                    elif self.graph.has_edge(y, z) and not (self.graph.has_edge(x, z) or self.graph.has_edge(z, x)):
                        self.graph.add_edge(z, x)
                        added_edge = (z, x)
                    if added_edge is not None:
                        self.skeleton[added_edge[0]][added_edge[1]]["conflict"] = False

        if self.save_folder is not None:
            self.save_graph(path=self.save_folder + "/02_chains.xml",
                            parent_cap=self.parent_cap,
                            eliminate_circles=True)

    def _find_common_parents(self):
        if len(self.v_scores) == 0:
            return

        if self.bon_correction:
            alpha_local = self.alpha / float(len(self.v_scores))
        else:
            alpha_local = self.alpha

        for row in self.v_scores:
            path = row[0:3]
            direct_scores = row[3:6]
            indirect_scores = row[6:9]
            for i in range(3):
                if np.isinf(direct_scores[i]) or np.isinf(indirect_scores[i]):
                    continue
                if direct_scores[i] < alpha_local and indirect_scores[i] > alpha_local:
                    z = path[i]
                    x, y = set(path) - {z}
                    added_edge = None
                    if not self.graph.has_edge(x, z) and not self.graph.has_edge(y, z):
                        self.graph.add_edge(z, x)
                        self.graph.add_edge(z, y)
                        added_edge = (z, y)
                    elif self.graph.has_edge(z, x) and not (self.graph.has_edge(y, z) or self.graph.has_edge(z, y)):
                        self.graph.add_edge(z, y)
                        added_edge = (z, y)
                    elif self.graph.has_edge(z, y) and not (self.graph.has_edge(z, x) or self.graph.has_edge(x, z)):
                        self.graph.add_edge(z, x)
                        added_edge = (z, x)
                    if added_edge is not None:
                        self.skeleton[added_edge[0]][added_edge[1]]["conflict"] = False

        if self.save_folder is not None:
            self.save_graph(path=self.save_folder + "/03_common_parents.xml",
                            parent_cap=self.parent_cap,
                            eliminate_circles=True)


if __name__ == "__main__":
    model_count = 25
    sample_counts = [2000, 5000, 10000]
    partition_size_list = [3, 2, 2]

    test_method(method_name="LCD",
                method=LCD,
                model_count=model_count,
                sample_counts=sample_counts,
                test_verbose=1,
                edge_generator=
                    StrictPartitionedRandomEdgeGenerator(partition_size_list=partition_size_list,
                                             possible_node_parent_counts_list=[
                                                 [0],
                                                 [2],
                                                 [1, 2]
                                             ],
                                             node_parent_count_probabilities_list=[
                                                 [1.],
                                                 [1.],
                                                 [0.6, 0.4]
                                             ]),
                value_count_generator=
                    RandomPartitionedValueCountGenerator(partition_size_list=partition_size_list,
                                                     possible_node_value_counts_list=[
                                                         [3],
                                                         [2],
                                                         [2]
                                                     ],
                                                     node_value_count_probabilities_list=[
                                                         [1.],
                                                         [1.],
                                                         [1.]
                                                     ]),
                # arguments for the LCD.fit() method:
                alpha=0.05,
                bonferroni_correction=False)
