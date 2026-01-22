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
import COHESION.Preprocessing.preprocessing_basic as pp
from COHESION.Utils import graph_construction, calcEnjoyment, time_call, normalize_scores


def getEdges(graph, u, community_node, t_obs):
    all_edges = list(graph.out_edges(u, data=True)) + list(graph.in_edges(u, data=True))
    uE, uOE = [], []
    user_pairs = set()

    for edge in all_edges:
        if int(edge[2]['timestamp']) <= t_obs:
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


def ATGS(graph, u, community_node, t_obs, rate, method):
    EI_value, SIT_value, CED_value = 0, 0, 0
    uE, uME, uOE = getEdges(graph, u, community_node, t_obs)

    # Calculate the Enjoyment Index (EI) of node u in subgraph H at time t_obs
    EI_value = calcEnjoyment(uE, t_obs, rate, method)

    # Calculate the Sentimental interaction tendency (SIT) of node i in subgraph H at time t_obs
    for _, edges in uME.items():
        SIT_value += calcEnjoyment(edges, t_obs, rate, method) 
    
    # Calculate the Comparative Enjoyment Degree(CED) of node i in subgraph H at time t_obs
    CED_value = EI_value - calcEnjoyment(uOE, t_obs, rate, method)

    return EI_value, SIT_value, CED_value


# Calculate the GI-S measures of subgraph H in graph G at time t_obs
def GIS(graph, community_node, t_obs):
    interaction_activities_num, total_activities_num = 0, 0
    GID_value, GIP_value = 0, 0
    nodes_num = len(community_node)

    subgraph = graph.subgraph(community_node)
    total_activities_num = subgraph.number_of_edges()
    interaction_activities_num = sum(1 for u, v in subgraph.edges() if u != v)

    if total_activities_num > 0:
        GIP_value = interaction_activities_num / total_activities_num
    
    if interaction_activities_num > 0:
        GID_value = interaction_activities_num / (nodes_num * (nodes_num - 1) * t_obs)

    return GIP_value, GID_value


def calc_cs(G, community_node, t_obs, method, rate, weights, LB_values, UB_values):

    EI_list, SIT_list, CED_list = [], [], []

    for u in tqdm.tqdm(community_node, desc="Calculating ATG-S measures for each user"):
        EI_u, SIT_u, CED_u = ATGS(G, u, community_node, t_obs, rate, method)
        EI_list.append(normalize_scores(EI_u, "EI", LB_values, UB_values))
        SIT_list.append(normalize_scores(SIT_u, "SIT", LB_values, UB_values))
        CED_list.append(normalize_scores(CED_u, "CED", LB_values, UB_values))
    
    EI_avg = round(np.mean(EI_list), 4)
    SIT_avg = round(np.mean(SIT_list), 4)
    CED_avg = round(np.mean(CED_list), 4)

    GIP, GID = GIS(G, community_node, t_obs)
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

        G = time_call("building the original graph", graph_construction, dataset_path, last_timestamps[dataset])
        LB_values, UB_values = time_call("calculating the boundaries", pp.findBounds, G, last_timestamps[dataset], decay_method, decay_rate)

        community_dir = community_path + dataset + "/"
        community_files = [community_dir + f for f in os.listdir(community_dir) if "Integrated" in f][0]
        time_spent_comp, time_spent_exp = [], []
        
        # Read each line of the community file, compute cohesiveness for each community
        with open(community_files, "r") as f:
            lines = f.readlines()
            results = []
            for line in tqdm.tqdm(lines):
                starttime = time.time()
                
                community = list(ast.literal_eval(line))
                S, MS = calc_cs(G, community, last_timestamps[dataset], decay_method, decay_rate, measure_weights, LB_values, UB_values)
                endtime = time.time()
                time_lapse_comp = endtime - starttime
                time_spent_comp.append(time_lapse_comp)

                starttime = time.time()
                explanation = generate_explanation(measure_weights, MS, S)
                endtime = time.time()
                time_lapse_exp = endtime - starttime
                time_spent_exp.append(time_lapse_exp) 

                results.append(f"{community}\t{MS}\t{S}\t{explanation}\t{time_lapse_comp}\t{time_lapse_exp}\n")
 
        avg_time_comp = np.mean(time_spent_comp)
        avg_time_exp = np.mean(time_spent_exp)

        print("Minimum values:", LB_values)
        print("Maximum values:", UB_values)
        print("Average time spent for cohesiveness computation(s):", avg_time_comp)
        print("Average time spent for explanation generation(s):", avg_time_exp)
        output_file = community_dir + dataset + "_communities_cohesiveness_IO_COHESION.txt"

        # Write the results to a txt file
        with open(output_file, "w") as f:
            f.write("Community\tMeasure Scores\tCohesiveness Score\tExplanation\tCompTime\tExpTime\n")
            for line in results:
                f.write(line)
        
        print(f"Results saved to {output_file}")