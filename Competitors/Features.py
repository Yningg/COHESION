"""
This script is written to generate explanations through informative features.
--> At the single node level: clustering coefficients; triangle participation; eigenvector centrality; and expansion.
--> At the node pair level: Jacaard similarity; Cosine similarity
"""

import ast
import os
import numpy as np
import ast
import os
import numpy as np
import networkx as nx
import tqdm
import sys
import sys
import math
import time

target_path = "./"
sys.path.append(target_path)
from COHESION.Utils import read_node_mapping
import time

target_path = "./"
sys.path.append(target_path)
from COHESION.Utils import read_node_mapping


def construct_graph(graph_path, node_mapping):
    undirected_G = nx.read_edgelist(graph_path, nodetype=int, data=False, create_using=nx.Graph()) # type: ignore
def construct_graph(graph_path, node_mapping):
    undirected_G = nx.read_edgelist(graph_path, nodetype=int, data=False, create_using=nx.Graph()) # type: ignore
    print(f"Graph info: {undirected_G.number_of_nodes()} nodes, {undirected_G.number_of_edges()} edges, density: {nx.density(undirected_G)}")
    undirected_G = nx.relabel_nodes(undirected_G, lambda x: node_mapping[int(x)])      
    undirected_G = nx.relabel_nodes(undirected_G, lambda x: node_mapping[int(x)])      
    return undirected_G

def construct_community(graph, community_nodes, node_mapping):
    community_nodes_mapped = [node_mapping[int(node)] for node in community_nodes]
    community = graph.subgraph(community_nodes_mapped).copy()
    return community

def construct_community(graph, community_nodes, node_mapping):
    community_nodes_mapped = [node_mapping[int(node)] for node in community_nodes]
    community = graph.subgraph(community_nodes_mapped).copy()
    return community

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
    n_nodes = list(G.nodes())
    n_nodes = list(G.nodes())
    node_degrees = dict(G.degree())

    for i in tqdm.trange(len(n_nodes)):
        for j in range(i+1, len(n_nodes)):
            jaccard = nx.jaccard_coefficient(G, [(n_nodes[i], n_nodes[j])])
            if len(pair_features['Jaccard']) <= 5000:
                pair_features['Jaccard']['nodes_{0}_{1}'.format(n_nodes[i], n_nodes[j])] = next(jaccard)[2]

            neighbors = len(list(nx.common_neighbors(G, n_nodes[i], n_nodes[j])))
            cosine = neighbors / (math.sqrt(node_degrees[n_nodes[i]]) * math.sqrt(node_degrees[n_nodes[j]]))

            if len(pair_features['Cosine Similarity']) <= 5000:
                pair_features['Cosine Similarity']['nodes_{0}_{1}'.format(n_nodes[i], n_nodes[j])] = cosine
    return pair_features     


def generate_explanation(graph, community):
    node_features = calc_node_features(graph, community)
    node_pair_features = calc_pair_features(community)

    # Order the node features by node id
    ordered_node_features = {node: node_features[node] for node in sorted(node_features.keys())}
    exp = "Node-level features:\n"

    for node, feats in ordered_node_features.items():
        if len(exp) < 5000:
            exp += f"Node {node}: {feats}\n"
        else:
            exp += "...\n"
            break
    exp += "\nNode-pair level features:\n"
    for feat_name, pairs in node_pair_features.items():
        if len(exp) < 5000:
            exp += f"\nFeature: {feat_name}, Values: {pairs}\n"
        else:
            exp += "...\n"
            break
    
    return exp
    

if __name__ == '__main__':
    # community_path = "./Datasets/Running_Example/community.txt" # Or "./Datasets/User_Study/community.txt"
    # graph_path = "./Datasets/Running_Example/graph.txt" # Or "./Datasets/User_Study/graph.txt"

    # Input parameters
    # dataset_list = ["BTW", "CC", "C26", "C144"]
    dataset_list = ["C144"]
    data_path = "./Datasets/OSNs/"
    community_path = "./Datasets/Communities/"

    for dataset in dataset_list:
        # Read node_mapping file
        node_mapping_path = "./Datasets/Node_Mapping/" + dataset + "_node_mapping.txt"
        node_mapping = read_node_mapping(node_mapping_path)

        graph = construct_graph(data_path + dataset + "_attributed.txt", node_mapping)

        community_dir = community_path + dataset + "/"
        community_files = [community_dir + f for f in os.listdir(community_dir) if "Integrated" in f][0]
        output_file = community_dir + dataset + "_communities_cohesiveness_Features.txt"

        time_spent_comp, time_spent_exp = [], []

        # Write the results to a txt file
        with open(output_file, "w") as fout:
            fout.write("Community\tExplanation\tCompTime\tExpTime\n")

            # Read each line of the community file, compute cohesiveness for each community
            with open(community_files, "r") as f:
                lines = f.readlines()
                for line in tqdm.tqdm(lines):
                    
                    starttime = time.time()
                    community = construct_community(graph, list(ast.literal_eval(line)), node_mapping)
                    endtime = time.time()
                    time_lapse_comp = endtime - starttime
                    time_spent_comp.append(time_lapse_comp)

                    starttime = time.time()
                    explanation = generate_explanation(graph, community)

                    endtime = time.time()
                    time_lapse_exp = endtime - starttime
                    time_spent_exp.append(time_lapse_exp)

                    fout.write(f"{community}\t{explanation}\t{time_lapse_comp}\t{time_lapse_exp}\n")
                    fout.flush()

        avg_time_comp = np.mean(time_spent_comp)
        avg_time_exp = np.mean(time_spent_exp)

        print("Average time spent for cohesiveness computation(s):", avg_time_comp)
        print("Average time spent for explanation generation(s):", avg_time_exp)
        
    
        print(f"Results saved to {output_file}")