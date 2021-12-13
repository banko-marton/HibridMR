import itertools

import networkx as nx
import pandas as pd

from pycausal.pycausal import pycausal as pc
from pycausal import search
from pycausal import prior as p
from cd_base import CDBase


class Tetrad_partioned(CDBase):
    def __init__(self, data: pd.DataFrame):
        CDBase.__init__(self, data=data)
        self.pc = pc()
        pc.start_vm(self, java_max_heap_size='5000M')
        self.tetrad = search.tetradrunner()

        self.df = data

    def get_layer(self, num, partition_count):
        extents = nx.utils.pairwise(itertools.accumulate((0,) + tuple(partition_count)))
        layers = [range(start, end) for start, end in extents]
        for ran in layers:
            if num in ran:
                return layers.index(ran)

    def fit(self, algoID, testID, partition_count, alpha=0.05):
        forbid = []
        partition_count.insert(0,0)
        # for idx in range(len(partition_count)-1):
        #     for i in range(partition_count[idx],partition_count[idx+1]):
        #         for j in range(partition_count[idx],partition_count[idx+1]):
        #             if i != j:
        #                 forbid.append(["node_" + str(i), "node_" + str(j)])

        prior = p.knowledge(forbiddirect=forbid)
        self.tetrad.run(algoId=algoID, dfs=self.df, testId=testID, dataType='discrete', priorKnowledge=prior, alpha = alpha)
        edges = []

        for e in self.tetrad.getEdges():
            e = e.split(' ')
            first = e[0]
            second = e[2]

            if e[1] == "-->" or e[1] == "o->":
                edges.append([first, second])
            elif e[1] == "<--" or e[1] == "<-o":
                edges.append([second, first])
            else:
                if int(e[0].split('_')[1]) < int(e[2].split('_')[1]):
                    edges.append([first, second])
                else:
                    edges.append([second, first])


        g = nx.DiGraph()
        for n in self.tetrad.getNodes():
            node_num = int(n.split('_')[1])
            g.add_node(n, layer=self.get_layer(node_num,partition_count))

        g.add_edges_from(edges)

        self.graph = g

        return g


# Algs and tests which can be used in Tetrad
    # alg: lingam
    # alg: r-skew
    # alg: fask
    # alg: gfci
    # alg: fas
    # alg: ftfc
    # alg: rfci-bsc
    # alg: ts-fci
    # alg: pc-all
    # alg: fges-mb
    # alg: imgs_cont
    # alg: ts-gfci
    # alg: ling
    # alg: glasso
    # alg: fofc
    # alg: skew
    # alg: ts-imgs
    # alg: fges
    # alg: multi-fask
    # alg: fci
    # alg: r3
    # alg: mbfs
    # alg: rfci
    # alg: mgm
    # alg: imgs_disc
    # ----
    # test
    # chi-square-test
    # test
    # dg-lr-test
    # test
    # fisher-z-test
    # test
    # mnlrlr-test
    # test
    # cci-test
    # test
    # d-sep-test
    # test
    # mvplr-test
    # test
    # kci-test
    # test
    # prob-test
    # test
    # bdeu-test
    # test
    # disc-bic-test
    # test
    # g-square-test
    # test
    # cg-lr-test