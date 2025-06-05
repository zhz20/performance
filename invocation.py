# -*- encoding = utf-8 -*-
"""
@description: a heuristic algorithm invoking the well-trained model to generate refactoring candidates
@date: 2024/2/12
@File : invocation.py
@Software : PyCharm
"""
import os

import dhg
import torch
from model import HGNNPLUS
import sys


def convert_hypergraph(graph):
    hypergraph = []
    for v in graph.v:
        edges = [v]
        for e in graph.e[0]:
            in_vertex, out_vertex = e
            if out_vertex == v:
                edges.append(in_vertex)
        hypergraph.append(edges)
    return dhg.Hypergraph(graph.v, hypergraph)


# Intra-class Dependency Graph Construction
def create_digraph(path):
    edges = []
    v = 0
    with open(path + '\\graph_node.csv', "r", encoding="utf-8") as gr:
        # View all members of a class
        lines = gr.readlines()
        v += len(lines)
    if not os.path.exists(path + '\\graph.csv'):
        return dhg.Graph(v, edges)
    with open(path + '\\graph.csv', "r", encoding="utf-8") as pr:
        for line in pr:
            line = line.strip('\n').split(',')
            edges.append([int(line[0]), int(line[1])])
    return dhg.Graph(v, edges)


def l1_mormalize(features):
    return torch.nn.functional.normalize(features, p=1, dim=1)


# codebert codegpt codet5 cotext graphcodebert plbart
def get_features(path, name):
    features = []
    with open(path + "\\embedding\\" + name + ".csv", 'r', encoding="utf-8") as gr:
        for line in gr:
            line = [float(x) for x in line.strip('\n').split(',')]
            features.append(line)
    features_tensor = torch.tensor(features)
    features_tensor = l1_mormalize(features_tensor)
    return features_tensor


class InvocationAlgorithm:
    def __init__(self, trained_model, hypergraph, features, fields_methods, num_extracted_classes):
        self.trained_model = trained_model
        self.hypergraph = hypergraph
        self.features = features
        self.fields_methods = fields_methods
        self.num_extracted_classes = num_extracted_classes
        self.clusters = [{fm} for fm in fields_methods]
        self.proximity_matrix = {}

    def calculate_likelihood_score(self, cluster_pair):
        link = torch.tensor([fm for cluster in cluster_pair for fm in cluster])
        output = model(self.features, self.hypergraph, link)
        return output  # Dummy value for demonstration

    def merge_clusters(self, cluster_i, cluster_j):
        # Merge two clusters
        return cluster_i.union(cluster_j)

    def model(self, cluster_pair):
        # Method to invoke the model for prediction
        return self.calculate_likelihood_score(cluster_pair)

    def max_value(self):
        max_score = 0
        max_cluster_pair = None
        for i in range(len(self.clusters)):
            for j in range(i + 1, len(self.clusters)):
                cluster_pair = (self.clusters[i], self.clusters[j])
                score = self.proximity_matrix[cluster_pair]
                if score > max_score:
                    max_score = score
                    max_cluster_pair = cluster_pair
        return max_score, max_cluster_pair

    def run_algorithm(self):
        for pair in self.all_pairs():
            self.proximity_matrix[pair] = self.model(pair)

        while len(self.clusters) > self.num_extracted_classes:
            score_h, (clus_i, clus_j) = self.max_value()
            if score_h < 0.5:
                return set()  # No classes can be extracted
            new_cluster = self.merge_clusters(clus_i, clus_j)
            self.clusters = [cluster for cluster in self.clusters if cluster != clus_i and cluster != clus_j]
            for cluster_k in self.clusters:
                self.proximity_matrix[(cluster_k, new_cluster)] = self.model((new_cluster, cluster_k))
            self.clusters.append(new_cluster)

        return self.clusters

    def all_pairs(self):
        return [(cluster1, cluster2) for i, cluster1 in enumerate(self.clusters) for cluster2 in self.clusters[i + 1:]]


if __name__ == "__main__":
    path = sys.argv[0]
    num_extracted_classes = sys.argv[1]
    # Example usage
    emb_type = 'codebert'
    features = get_features(path, emb_type)
    g = create_digraph(path)
    hg = convert_hypergraph(g)
    # Assuming hg is properly initialized with v, e, and feature tensors
    fields_methods = hg.v
    # load well-trained model
    model = HGNNPLUS(768, 256, 16)
    state_dict = torch.load("model/hecs.pth")
    model.load_state_dict(state_dict)

    algorithm = InvocationAlgorithm(model, hg, features, fields_methods, num_extracted_classes)
    extracted_classes = algorithm.run_algorithm()
    print("Extracted Classes:", extracted_classes)
