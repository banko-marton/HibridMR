import networkx as nx
import pandas as pd
import numpy as np
import os
import time
from itertools import combinations

from cd_base import CDBase
from data.generate_test_data import generate_data, RandomEdgeGenerator, RandomValueCountGenerator


def compare_models(true_model: nx.DiGraph, predicted_model: nx.DiGraph):
    assert set(true_model.nodes) == set(predicted_model.nodes)
    all_pairs = list(combinations(true_model.nodes, 2))
    TP, TN, FP, FN = 0, 0, 0, 0
    correct, unnecessary, absent, reverse = 0, 0, 0, 0
    for n1, n2 in all_pairs:
        for x, y in [(n1, n2), (n2, n1)]:
            if true_model.has_edge(x, y):
                if predicted_model.has_edge(x, y):
                    TP += 1
                    correct += 1
                elif predicted_model.has_edge(y, x):
                    FN += 1
                    reverse += 1
                else:
                    FN += 1
                    absent += 1
            else:
                if predicted_model.has_edge(x, y):
                    FP += 1
                    unnecessary += 1
                else:
                    TN += 1
    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN,
            "correct": correct, "unnecessary": unnecessary,
            "absent": absent, "reverse": reverse}


def get_scores(true_model: nx.DiGraph, predicted_model: nx.DiGraph):
    result = compare_models(true_model=true_model, predicted_model=predicted_model)

    original_edge_count = result["TP"] + result["FN"]
    if original_edge_count > 0:
        recall = result["correct"] / original_edge_count
        original_absent_rate = result["absent"] / original_edge_count
        original_reverse_rate = result["reverse"] / original_edge_count
    else:
        recall = 0.
        original_absent_rate = 0.
        original_reverse_rate = 0.

    predicted_edge_count = result["TP"] + result["FP"]
    if predicted_edge_count > 0:
        precision = result["correct"] / predicted_edge_count
        predicted_unnecessary_rate = (result["unnecessary"] - result["reverse"]) / predicted_edge_count
        predicted_reverse_rate = result["reverse"] / predicted_edge_count
    else:
        precision = 0.
        predicted_unnecessary_rate = 0.
        predicted_reverse_rate = 0.

    return [recall, original_absent_rate, original_reverse_rate,
            precision, predicted_unnecessary_rate, predicted_reverse_rate]


def perform_test(method: type(CDBase), data_path: str, **kwargs):
    data = pd.read_csv(data_path)
    model = method(data=data)
    model.fit(**kwargs)
    return model.graph


def test_method(method_name: str, method: type(CDBase), model_count, sample_counts,
                edge_generator=None, value_count_generator=None,
                test_verbose=0, **kwargs):

    if edge_generator is None:
        edge_generator = RandomEdgeGenerator(node_count=60,
                                             possible_node_parent_counts=[1, 2, 3, 4, 5],
                                             node_parent_count_probabilities=[0.1, 0.2, 0.4, 0.2, 0.1])

    if value_count_generator is None:
        value_count_generator = RandomValueCountGenerator(node_count=60,
                                                          possible_node_value_counts=[2, 3, 4],
                                                          node_value_count_probabilities=[0.3, 0.4, 0.3])

    generate_data(edge_generator=edge_generator,
                  value_count_generator=value_count_generator,
                  model_count=model_count,
                  sample_counts=sample_counts,
                  data_folder="data",
                  overwrite=False)

    results_base_folder = "results/" + method_name
    if not os.path.isdir(results_base_folder):
        os.makedirs(results_base_folder)

    results = []
    execution_times = []
    for sample_count in sample_counts:
        if test_verbose:
            print("Testing on sample count " + str(sample_count) + "...")
        base_folder = "data/partitioned_models_node_" + str(edge_generator.node_count) + "_sample_" + str(sample_count)
        result_model_base_folder = results_base_folder + "/partitioned_models_node_" + str(edge_generator.node_count) + "_sample_" + str(sample_count)
        if not os.path.isdir(result_model_base_folder):
            os.makedirs(result_model_base_folder)
        sc_results = []
        sc_execution_times = []
        for model_index in range(model_count):
            data_path = base_folder + "/data_" + str(model_index) + ".csv"
            graph_path = base_folder + "/model_" + str(model_index) + ".xml"
            predicted_graph_path = result_model_base_folder + "/model_" + str(model_index) + ".xml"
            start_time = time.time()
            predicted_graph = perform_test(method=method, data_path=data_path, **kwargs)
            original_graph = nx.read_graphml(graph_path)
            sc_results.append(get_scores(true_model=original_graph, predicted_model=predicted_graph))
            exec_time = time.time() - start_time
            sc_execution_times.append(exec_time)
            nx.write_graphml_xml(predicted_graph, predicted_graph_path)
            if test_verbose:
                print(data_path + " calculated in %.2f seconds." % exec_time)
        sc_results = np.array(sc_results)
        results.append(sc_results.mean(axis=0))
        execution_times.append(np.mean(sc_execution_times))
    results = np.array(results)
    sample_counts = np.array(sample_counts)
    sample_counts = sample_counts.reshape((len(sample_counts), 1))
    execution_times = np.array(execution_times)
    execution_times = execution_times.reshape((len(sample_counts), 1))
    results = np.concatenate((sample_counts, results, execution_times), axis=1)
    results_df = pd.DataFrame(data=results,
                              columns=["sample_count",
                                       "recall", "original_absent_rate", "original_reverse_rate",
                                       "precision", "predicted_unnecessary_rate", "predicted_reverse_rate",
                                       "execution_time"])

    result_file_path = results_base_folder + '/' + method_name + "_performance.csv"
    if test_verbose:
        print("Writing results to: " + result_file_path)
    results_df.to_csv(result_file_path, index=False)
