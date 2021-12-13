import itertools
import os
import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt

import LCD
import MR
from tetrad import Tetrad_partioned


class ModelData:
    def __init__(self, typeID, gene_part, exp_part, out_part):
        self.typeID = typeID
        self.gene_part = gene_part
        self.exp_part = exp_part
        self.out_part = out_part

def get_layer(num, partition_count):
    extents = nx.utils.pairwise(itertools.accumulate((0,) + tuple(partition_count)))
    layers = [range(start, end) for start, end in extents]
    for ran in layers:
        if num in ran:
            return layers.index(ran)

if __name__ == "__main__":
    mr = MR.MR()

    models = [
        ModelData(1, [0,1,2], [3], [4]),
        ModelData(2, [0,1,2], [3,4], [5]),
        ModelData(3, [0,1,2], [3], [4,5]),
        ModelData(4, [0,1,2], [3,4], [5,6])
    ]

    for model in models:
        for samplesize in [100,200,500,1000,2000,5000,10000]:
            source_dir = f"szakdoga_data/model_{model.typeID}/{samplesize}"
            data_path = f"{source_dir}/data.csv"

            # Calculating odds ratios
            mr.calc_or(datasource=source_dir,
                       trait_node_id_range=model.exp_part + model.out_part,
                       gene_node_id_range=model.gene_part)

            if not os.path.exists(f"{source_dir}/Causal_results"):
                os.makedirs(f"{source_dir}/Causal_results")

            if not os.path.exists(f"{source_dir}/MR_results/Simple_MR_results"):
                os.makedirs(f"{source_dir}/MR_results/Simple_MR_results")

            for gene_id in model.gene_part:
                for exp_id in model.exp_part:
                    for out_id in model.out_part:
                        mr.calc_beta(gene_node_id=gene_id,
                                            exposure_id=exp_id,
                                            outcome_id=out_id,
                                            save_dir= f"{source_dir}/MR_results/Simple_MR_results",
                                            stat_data_source=f"{source_dir}/MR_results",
                                            modelfile=f"{source_dir}/model.xml")

            # calculating causal discovery algorithms
            for alg in ["LCD", "pc-all", "fci", "rfci"]:
                for test in ["g-square-test", "chi-square-test"]:
                    module = Tetrad_partioned(pd.read_csv(data_path))
                    subset_sizes = [len(model.gene_part), len(model.exp_part), len(model.out_part)]
                    subset_color = [
                        "indianred",
                        "skyblue",
                        "gold"
                    ]

                    if not os.path.exists(f"{source_dir}/MR_results/{alg}_{test}"):
                        os.makedirs(f"{source_dir}/MR_results/{alg}_{test}")

                    # LCD alg is in environment
                    if(alg == "LCD"):
                        result_graph = LCD.LCD(save_folder=f"{source_dir}/Causal_results",
                                            data=pd.read_csv(data_path))
                        result_graph.fit(alpha=0.05, bonferroni_correction=False, verbose=0)

                        try:
                            result_graph.skeleton = nx.read_graphml(f"{source_dir}/Causal_results/03_common_parents.xml")
                        except FileNotFoundError:
                            print("LCD could not find strong enough bonds, printing skeleton")

                        for node in result_graph.skeleton.nodes:
                            node_num = int(node.split('_')[1])
                            result_graph.skeleton.nodes[node]["layer"] = get_layer(node_num, subset_sizes)

                        color = [subset_color[data["layer"]] for v, data in result_graph.skeleton.nodes(data=True)]
                        pos = nx.multipartite_layout(result_graph.skeleton, subset_key="layer")
                        plt.figure(figsize=(5, 5))
                        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.6, top=0.8)
                        nx.draw(result_graph.skeleton, pos, node_color=color, node_size=2000, with_labels=True)
                        plt.axis("equal")
                        plt.savefig(f"{source_dir}/Causal_results/LCD_graph.png", format="PNG", dpi=300)

                        edges = []
                        for edge in result_graph.skeleton.edges:
                            src = edge[0].split('_')[1]
                            dest = edge[1].split('_')[1]
                            edges.append((int(src), int(dest)))

                        for gene_id in model.gene_part:
                            for exp_id in model.exp_part:
                                for out_id in model.out_part:
                                    if((gene_id, exp_id) in edges and (exp_id, out_id) in edges):
                                        mr.calc_beta(gene_node_id=gene_id,
                                                            exposure_id=exp_id,
                                                            outcome_id=out_id,
                                                            save_dir= f"{source_dir}/MR_results/{alg}_{test}",
                                                            stat_data_source=f"{source_dir}/MR_results",
                                                            modelfile=f"{source_dir}/model.xml")


                    # We use tetrad for all other algs
                    else:
                        result_graph = module.fit(algoID=alg,
                                                  testID=test,
                                                  partition_count=subset_sizes)

                        color = [subset_color[data["layer"] - 1] for v, data in result_graph.nodes(data=True)]
                        pos = nx.multipartite_layout(result_graph, subset_key="layer")
                        plt.figure(figsize=(5, 5))
                        nx.draw(result_graph, pos, node_color=color, node_size=2000, with_labels=True)
                        plt.axis("equal")
                        plt.savefig(f"{source_dir}/Causal_results/{alg}_{test}.png", format="PNG", dpi=300)

                        with open(f"{source_dir}/Causal_results/{alg}_{test}.txt", "w") as outfile:
                            outfile.write(str(module.tetrad.getTetradGraph()))

                        edges = []
                        for e in module.tetrad.getEdges():
                            e = e.split(' ')
                            first = e[0]
                            second = e[2]

                            if e[1] == "-->" or e[1] == "o->":
                                edges.append((int(first.split('_')[1]), int(second.split('_')[1])))

                        # If we have a gene - exp - out edges, both
                        # Using odds ratios we measure beta effects
                        for gene_id in model.gene_part:
                            for exp_id in model.exp_part:
                                for out_id in model.out_part:
                                    if((gene_id, exp_id) in edges and (exp_id, out_id) in edges):
                                        mr.calc_beta(gene_node_id=gene_id,
                                                            exposure_id=exp_id,
                                                            outcome_id=out_id,
                                                            save_dir= f"{source_dir}/MR_results/{alg}_{test}",
                                                            stat_data_source=f"{source_dir}/MR_results",
                                                            modelfile=f"{source_dir}/model.xml")

    print("End of run")
