import sys
import networkx as nx
import tqdm
import os
import time
import numpy as np
import ast
from collections import defaultdict


target_path = "./"
sys.path.append(target_path)
from COHESION.Explanation_generation import generate_explanation
import COHESION.FindBounds.findBounds_basic as fb
import COHESION.Utils as utils



def getEdges(graph, u, community_node, cur_t):
    all_edges = list(graph.out_edges(u, data=True)) + list(graph.in_edges(u, data=True))
    uE, uOE = [], []
    user_pairs = set()

    for edge in all_edges:
        if int(edge[2]['timestamp']) <= cur_t:
            if edge[1] in community_node and edge[0] in community_node:
                uE.append((edge[0], edge[1], int(edge[2]['timestamp']), int(edge[2]['sentiment'])))
                user_pairs.add((edge[0], edge[1]))
            else:
                uOE.append((int(edge[2]['timestamp']), int(edge[2]['sentiment'])))

    uE = sorted(uE, key=lambda x: x[2])  # Sort by timestamp
    
    uME = defaultdict(list)
    mutual_pairs = {(min(pair[0], pair[1]), max(pair[0], pair[1])) for pair in user_pairs if (pair[1], pair[0]) in user_pairs and pair[0] != pair[1]}

    if mutual_pairs:
        for (u, v, t, s) in uE:
            key = (u, v) if u < v else (v, u)
            if key in mutual_pairs:
                uME[key].append((t, s))

    uE = [(t, s) for (_, _, t, s) in uE]

    return uE, uME, uOE


def ATGS(graph, u, community_node, cur_t, rate, method):
    EI_value, SIT_value, CED_value = 0, 0, 0
    uE, uME, uOE = getEdges(graph, u, community_node, cur_t)

    # Calculate the Enjoyment Index (EI) of node u in subgraph H at time cur_t
    EI_value = utils.calcEnjoyment(uE, cur_t, rate, method)

    # Calculate the Sentimental interaction tendency (SIT) of node i in subgraph H at time cur_t
    for pair, edges in uME.items():
        SIT_value += utils.calcEnjoyment(edges, cur_t, rate, method)
    
    
    # Calculate the Comparative Enjoyment Degree(CED) of node i in subgraph H at time cur_t
    CED_value = EI_value - utils.calcEnjoyment(uOE, cur_t, rate, method)

    return EI_value, SIT_value, CED_value


# Calculate the GI-S measures of subgraph H in graph G at time cur_t
def GIS(graph, community_node, cur_t):
    interaction_activities_num, total_activities_num = 0, 0
    GID_value, GIP_value = 0, 0
    nodes_num = len(community_node)

    subgraph = graph.subgraph(community_node)
    total_activities_num = subgraph.number_of_edges()
    interaction_activities_num = sum(1 for u, v in subgraph.edges() if u != v)

    if total_activities_num > 0:
        GIP_value = interaction_activities_num / total_activities_num
    
    if interaction_activities_num > 0:
        GID_value = interaction_activities_num / (nodes_num * (nodes_num - 1) * cur_t)

    return GIP_value, GID_value


def normalize_scores(score, measure, LB_values, UB_values):
    min_val = LB_values[measure]
    max_val = UB_values[measure]
    if max_val == min_val: 
        return 0.0 
    return (score - min_val) / (max_val - min_val)



def calc_cs(G, community_node, cur_t, method, rate, weights, LB_values, UB_values):

    EI_list, SIT_list, CED_list = [], [], []

    for u in tqdm.tqdm(community_node, desc="Calculating ATG-S measures for each user"):
        EI_u, SIT_u, CED_u = ATGS(G, u, community_node, cur_t, rate, method)
        EI_list.append(normalize_scores(EI_u, "EI", LB_values, UB_values))
        SIT_list.append(normalize_scores(SIT_u, "SIT", LB_values, UB_values))
        CED_list.append(normalize_scores(CED_u, "CED", LB_values, UB_values))
    
    EI_avg = round(np.mean(EI_list), 4)
    SIT_avg = round(np.mean(SIT_list), 4)
    CED_avg = round(np.mean(CED_list), 4)

    GIP, GID = GIS(G, community_node, cur_t)
    GIP = round(normalize_scores(GIP, "GIP", LB_values, UB_values), 4)
    GID = round(normalize_scores(GID, "GID", LB_values, UB_values), 4)
    measure_scores = np.array([float(EI_avg), float(SIT_avg), float(CED_avg), GIP, GID])

    score = round(np.dot(measure_scores, weights), 4)

    return score, measure_scores


if __name__ == '__main__':
    # Input parameters
    decay_method = "exp"
    decay_rate = 0.0001
    measure_weights = np.array([1/3, 1/3, 1/3, 1/2, 1/2])
    dataset_list = ["BTW", "CC", "C26", "C144"]
    data_path = "./Datasets/OSNs/"
    community_path = "./Datasets/Communities/"
    last_timestamps = {"BTW": 1506315747, "CC": 1643673425, "C26": 1672531185, "C144": 1672531150}


    for dataset in dataset_list:
        dataset_path = data_path + dataset + "_attributed.txt"

        G = utils.graph_construction(dataset_path)
        LB_values, UB_values, time_spent = fb.findBounds(G, last_timestamps[dataset], decay_method, decay_rate)

        community_dir = community_path + dataset + "/"
        community_files = [community_dir + f for f in os.listdir(community_dir) if "Integrated" in f][0]

        # Read each line of the community file, compute cohesiveness for each community
        with open(community_files, "r") as f:
            lines = f.readlines()
            results = []
            for line in tqdm.tqdm(lines):
                community = list(ast.literal_eval(line))
                S, MS = calc_cs(G, community, last_timestamps[dataset], decay_method, decay_rate, measure_weights, LB_values, UB_values)
                explanation = generate_explanation(measure_weights, MS, S)

                results.append(f"{community}\t{MS}\t{S}\t{explanation}\n")  

        print("Minimum values:", LB_values)
        print("Maximum values:", UB_values)
        print("Time spent for boundary explanation (s):", time_spent)
        output_file = community_dir + dataset + "_communities_cohesiveness_IO_COHESION.txt"

        # Write the results to a txt file
        with open(output_file, "w") as f:
            f.write("Community\tMeasure Scores\tCohesiveness Score\tExplanation\n")
            for line in results:
                f.write(line)
        
        print(f"Results saved to {output_file}")