import random
from functools import reduce
import networkx as nx
import os
import pandas as pd
from multiprocessing import Pool
import numpy as np

from pomegranate import DiscreteDistribution, ConditionalProbabilityTable, State, BayesianNetwork, utils


def convert_index_to_value(index, value_counts):
    result = [0 for _ in range(len(value_counts))]
    carry = index
    for i in range(len(value_counts) - 1, -1, -1):
        result[i] = carry % value_counts[i]
        carry = carry // value_counts[i]
        if carry == 0:
            break
    return tuple(result)


def convert_value_to_index(value, value_counts):
    exponents = [reduce(lambda x, y: x*y, [value_counts[j] for j in range(i + 1, len(value_counts))]) for i in range(len(value_counts) - 1)]
    exponents.append(1)
    return sum([value[i] * exponents[i] for i in range(len(value))])


class EdgeGenerator:
    def __init__(self, node_count: int, possible_node_parent_counts: list, node_parent_count_probabilities: list):
        assert len(possible_node_parent_counts) == len(node_parent_count_probabilities)
        self.node_count = node_count
        self.possible_node_parent_counts = possible_node_parent_counts
        self.node_parent_count_probabilities = node_parent_count_probabilities

    def __call__(self, *args, **kwargs) -> list:
        raise NotImplementedError("The __call__ method has to be implemented!")


class RandomEdgeGenerator(EdgeGenerator):
    def __call__(self):
        edges = [(0, 2), (1, 2)]
        for i in range(3, self.node_count):
            parent_count = random.choices(self.possible_node_parent_counts,
                                          weights=self.node_parent_count_probabilities)[0]
            if parent_count > i:
                parent_count = i
            parents = sorted(random.sample(list(range(i)), parent_count))
            for parent in parents:
                edges.append((parent, i))

        return edges


class StrictPartitionedRandomEdgeGenerator(EdgeGenerator):
    def __init__(self, partition_size_list: list, possible_node_parent_counts_list: list,
                 node_parent_count_probabilities_list: list):
        EdgeGenerator.__init__(self, node_count=sum(partition_size_list),
                               possible_node_parent_counts=possible_node_parent_counts_list[0],
                               node_parent_count_probabilities=node_parent_count_probabilities_list[0])
        assert len(partition_size_list) == len(possible_node_parent_counts_list)
        assert len(partition_size_list) == len(node_parent_count_probabilities_list)
        self.partition_size_list = partition_size_list
        self.possible_node_parent_counts_list = possible_node_parent_counts_list
        self.node_parent_count_probabilities_list = node_parent_count_probabilities_list

    def __call__(self):
        partition_borders = [0]
        partition_borders.extend([sum(self.partition_size_list[:(i + 1)]) for i in range(len(self.partition_size_list))])
        edges = []
        for i in range(1, len(partition_borders) - 1):
            for j in range(partition_borders[i], partition_borders[i + 1]):
                parent_count = random.choices(self.possible_node_parent_counts_list[i],
                                              weights=self.node_parent_count_probabilities_list[i])[0]
                possible_parents = list(range(partition_borders[i - 1], partition_borders[i]))
                possible_parents_without_children = possible_parents.copy()
                for edge in edges:
                    try:
                        possible_parents_without_children.remove(edge[0])
                    except ValueError:
                        pass
                if parent_count > len(possible_parents):
                    parent_count = len(possible_parents)
                if len(possible_parents_without_children) > 0:
                    if parent_count <= len(possible_parents_without_children):
                        possible_parents = possible_parents_without_children
                    else:
                        possible_parents_with_children = possible_parents.copy()
                        for parent in possible_parents_without_children:
                            possible_parents_with_children.remove(parent)
                        possible_parents = possible_parents_without_children
                        possible_parents.extend(random.sample(possible_parents_with_children, parent_count - len(possible_parents)))
                parents = sorted(random.sample(possible_parents, parent_count))
                for parent in parents:
                    edges.append((parent, j))

        return edges


class ValueCountGenerator:
    def __init__(self, node_count: int, possible_node_value_counts: list, node_value_count_probabilities: list):
        assert len(possible_node_value_counts) == len(node_value_count_probabilities)
        self.node_count = node_count
        self.possible_node_value_counts = possible_node_value_counts
        self.node_value_count_probabilities = node_value_count_probabilities

    def __call__(self, *args, **kwargs) -> list:
        raise NotImplementedError("The __call__ method has to be implemented!")


class RandomValueCountGenerator(ValueCountGenerator):
    def __call__(self):
        return [random.choices(self.possible_node_value_counts,
                               weights=self.node_value_count_probabilities)[0] for _ in range(self.node_count)]


class RandomPartitionedValueCountGenerator(ValueCountGenerator):
    def __init__(self, partition_size_list: list, possible_node_value_counts_list: list,
                 node_value_count_probabilities_list: list):
        ValueCountGenerator.__init__(self, node_count=sum(partition_size_list),
                                     possible_node_value_counts=possible_node_value_counts_list[0],
                                     node_value_count_probabilities=node_value_count_probabilities_list[0])
        assert len(partition_size_list) == len(possible_node_value_counts_list)
        assert len(partition_size_list) == len(node_value_count_probabilities_list)
        self.partition_size_list = partition_size_list
        self.possible_node_value_counts_list = possible_node_value_counts_list
        self.node_value_count_probabilities_list = node_value_count_probabilities_list

    def __call__(self):
        value_counts = []
        for i in range(len(self.partition_size_list)):
            value_counts.extend([random.choices(self.possible_node_value_counts_list[i],
                                                weights=self.node_value_count_probabilities_list[i])[0] for _ in range(self.partition_size_list[i])])
        return value_counts


def get_test(edge_generator: type(EdgeGenerator), value_count_generator: type(ValueCountGenerator), sample_count: int = 1000):
    edges = edge_generator()

    parent_rows = [[] for i in range(edge_generator.node_count)]
    for edge in edges:
        parent_rows[edge[1]].append(edge[0])
    value_counts = value_count_generator()
    probability_dicts = []
    for i in range(len(parent_rows)):
        if len(parent_rows[i]) == 0:
            combination_count = 1
        else:
            combination_count = reduce(lambda x, y: x*y, [value_counts[parent] for parent in parent_rows[i]])
        conditional_probabilities = []
        for _ in range(combination_count):
            probabilities = []
            for k in range(value_counts[i]):
                probabilities.append(random.random())
            prob_norm = sum(probabilities)
            probabilities = [(prob / prob_norm) for prob in probabilities]
            conditional_probabilities.append(probabilities)
        if len(parent_rows[i]) == 0:
            probability_dicts.append(conditional_probabilities[0])
        else:
            combinations = [convert_index_to_value(index, [value_counts[parent] for parent in parent_rows[i]]) for index in range(combination_count)]
            prob_dict = {}
            for combination, probabilities in zip(combinations, conditional_probabilities):
                prob_dict[combination] = probabilities
            probability_dicts.append(prob_dict)


    distributions = []
    for i in range(len(parent_rows)):
        if len(parent_rows[i]) == 0:
            prob_dict = {index: prob for index, prob in enumerate(probability_dicts[i])}
            dist = DiscreteDistribution(prob_dict)
            distributions.append(dist)
        else:
            combination_probs = []
            for parent_value_comb in probability_dicts[i].keys():
                for value in range(len(probability_dicts[i][parent_value_comb])):
                    row = [element for element in parent_value_comb]
                    row.append(value)
                    row.append(probability_dicts[i][parent_value_comb][value])
                    combination_probs.append(row)
            parent_distributions = [distributions[parent_index] for parent_index in parent_rows[i]]
            dist = ConditionalProbabilityTable(combination_probs, parent_distributions)
            distributions.append(dist)

    nodes = []
    for i in range(len(distributions)):
        node = State(distributions[i], str(i))
        nodes.append(node)

    model = BayesianNetwork()

    for i in range(len(nodes)):
        model.add_node(nodes[i])

    for i in range(len(parent_rows)):
        for parent_index in parent_rows[i]:
            model.add_edge(nodes[parent_index], nodes[i])

    model.bake()

    graph = nx.DiGraph()
    graph.add_nodes_from([str(i) for i in range(edge_generator.node_count)])
    for i, parent_list in enumerate(parent_rows):
        for parent in parent_list:
            graph.add_edge(str(parent), str(i))
    graph = nx.relabel_nodes(graph, {str(i): "node_" + str(i) for i in range(edge_generator.node_count)})

    # Check if there are no isolated nodes in the model:
    node_degrees = np.array([degree for (node, degree) in graph.to_undirected().degree])
    assert node_degrees.min() > 0

    samples = model.sample(n=sample_count)

    return graph, samples


def run_and_save(edge_generator: type(EdgeGenerator), value_count_generator: type(ValueCountGenerator),
                 sample_count, base_folder, model_index, overwrite=True):
    graph_path = base_folder + "/model_" + str(model_index) + ".xml"
    data_path = base_folder + "/data_" + str(model_index) + ".csv"
    if overwrite or (not os.path.isfile(graph_path)) or (not os.path.isfile(data_path)):
        graph, samples = get_test(edge_generator=edge_generator, value_count_generator=value_count_generator,
                                  sample_count=sample_count)
        samples = pd.DataFrame(data=samples, columns=["node_" + str(index) for index in range(edge_generator.node_count)])
        nx.write_graphml_xml(graph, graph_path)
        samples.to_csv(data_path, index=False)


def generate_data(edge_generator: type(EdgeGenerator), value_count_generator: type(ValueCountGenerator),
                  model_count, sample_counts, data_folder, overwrite=True, n_jobs=6):
    inputs = []
    for sample_count in sample_counts:
        if 'Partitioned' in str(type(edge_generator)):
            base_folder = data_folder + "/partitioned_models_node_" + str(edge_generator.node_count) + "_sample_" + str(sample_count)
        else:
            base_folder = data_folder + "/models_node_" + str(edge_generator.node_count) + "_sample_" + str(sample_count)
        if not os.path.isdir(base_folder):
            os.makedirs(base_folder)

        for i in range(model_count):
            inputs.append((edge_generator, value_count_generator, sample_count, base_folder, i, overwrite))

    with Pool(processes=n_jobs) as pool:
        pool.starmap(run_and_save, inputs)


if __name__ == "__main__":
    n_jobs = 6

    # For the Random generation:
    node_count = 7

    # For the partitioned Random generation:
    partition_size_list = [3, 1, 1]

    model_count = 1
    sample_counts = [10000]

    edge_generators = [
        # RandomEdgeGenerator(node_count=node_count,
        #                     possible_node_parent_counts=[1, 2, 3, 4, 5],
        #                     node_parent_count_probabilities=[0.1, 0.2, 0.4, 0.2, 0.1]),
        StrictPartitionedRandomEdgeGenerator(partition_size_list=partition_size_list,
                                             possible_node_parent_counts_list=[
                                                 [0],
                                                 [3],
                                                 [1]
                                             ],
                                             node_parent_count_probabilities_list=[
                                                 [1.],
                                                 [1.],
                                                 [1.]
                                             ])
    ]

    value_count_generators = [
        # RandomValueCountGenerator(node_count=node_count,
        #                           possible_node_value_counts=[2, 3, 4],
        #                           node_value_count_probabilities=[0.3, 0.4, 0.3]),
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
                                             ])
    ]

    for edge_generator, value_count_generator in zip(edge_generators, value_count_generators):
        generate_data(edge_generator=edge_generator, value_count_generator=value_count_generator,
                      model_count=model_count, sample_counts=sample_counts, data_folder=os.getcwd(),
                      overwrite=True, n_jobs=n_jobs)
