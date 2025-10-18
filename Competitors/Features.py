"""
Generate explanations through informative features.
--> At the single node level: clustering coefficients; triangle participation; eigenvector centrality; and expansion.
--> At the node pair level: Jacaard similarity; Cosine similarity
"""

import pandas as pd
import networkx as nx
import tqdm
import math
import matplotlib.pyplot as plt


def construct_graph(graph_path):
    undirected_G = nx.read_edgelist(graph_path, nodetype=int, data=(('timestamp', str), ('sentiment', str)), create_using=nx.Graph()) # type: ignore

    # Delete attributes for simplicity
    for _, _, d in undirected_G.edges(data=True):
        d.clear()
        
    return undirected_G


"""
Node-level features
"""
def calc_node_features(G, H):
    features = {}
    w = H.number_of_nodes()

    node_clustering_coefficients = nx.clustering(H)
    node_eigenvector = nx.eigenvector_centrality(H, max_iter=1000)

    for node in H.nodes():
        e_out = G.degree(node) - H.degree(node)
        expansion = e_out / w
        clustering_coefficient = node_clustering_coefficients[node]
        eigenvector_centrality = node_eigenvector[node]

        tp_nods = []
        neighbors = list(H.neighbors(node))
        for nbr_nod in neighbors:
            if nbr_nod not in tp_nods:
                for nbr_nod_2 in neighbors:
                    if H.has_edge(nbr_nod, nbr_nod_2):
                        tp_nods.extend((nbr_nod, nbr_nod_2))
                        break
        triangle_participation = len(list(set(tp_nods)))/w
        

        features[node] = {
            'clustering_coefficient': clustering_coefficient,
            'triangle_participation': triangle_participation,
            'eigenvector_centrality': eigenvector_centrality,
            'expansion': expansion
        }
    return features


"""
Node-pair level features
"""
def calc_pair_features(G):
    pair_features = {'Jaccard': {}, 'Cosine Similarity': {}}
    n_nodes = G.number_of_nodes()
    node_degrees = dict(G.degree())

    for i in tqdm.trange(n_nodes):
        for j in range(i+1, n_nodes):

            jaccard = nx.jaccard_coefficient(G, [(i,j)])
            pair_features['Jaccard']['nodes_{0}_{1}'.format(i,j)] = next(jaccard)[2]

            neighbors = len(list(nx.common_neighbors(G, i, j)))
            cosine = neighbors / (math.sqrt(node_degrees[i]) * math.sqrt(node_degrees[j]))
            pair_features['Cosine Similarity']['nodes_{0}_{1}'.format(i,j)] = cosine

    return pair_features     

if __name__ == "__main__":

    community_path = "./Dataset/User_Study/community.txt"
    graph_path = "./Dataset/User_Study/graph.txt"

    community = construct_graph(community_path)
    graph = construct_graph(graph_path)

    node_features = calc_node_features(graph, community)
    node_pair_features = calc_pair_features(community)

    # Order the node features by node id
    ordered_node_features = {node: node_features[node] for node in sorted(node_features.keys())}
    print("Node-level features:")
    for node, feats in ordered_node_features.items():
        print(f"Node {node}: {feats}")
    
    print("\nNode-pair level features:")
    for feat_name, pairs in node_pair_features.items():
        print(f"\nFeature: {feat_name}, Values: {pairs}")