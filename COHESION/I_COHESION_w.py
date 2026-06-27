import sys
import tqdm
import os
import time
import numpy as np
import ast

target_path = "./"
sys.path.append(target_path)

from COHESION.Explanation_generation import generate_explanation
import COHESION.Preprocessing.preprocessing_index as pp_i
from COHESION.Utils import calcEnjoyment, normalize_scores, preprocess_dataset, read_node_mapping


def ATGS(index, u, community_node, t_obs, method, rate):
    
    EI_value, SIT_value, CED_value = 0, 0, 0
    total_neighbors = index["NI"][u][0] | index["NI"][u][1]
    in_community_neighbors = total_neighbors & set(community_node)
    out_community_neighbors = total_neighbors - in_community_neighbors
    mutual_neighbors = index["NI"][u][1] & set(community_node)

    uE = pp_i.getEdges(index["PI"], u, in_community_neighbors)
    EI_value = calcEnjoyment(uE, t_obs, method, rate)
    
    SIT_value = 0
    uME = pp_i.getPairEdges(index["PI"], u, mutual_neighbors)
    for _, edges in uME.items():
        SIT_value += calcEnjoyment(edges, t_obs, method, rate)

    uOE = pp_i.getEdges(index["PI"], u, out_community_neighbors)
    CED_value = EI_value - calcEnjoyment(uOE, t_obs, method, rate)

    return EI_value, SIT_value, CED_value


# Calculate the GI-S measures of subgraph H in graph G at time t_obs
def GIS(PI, community_node, t_obs):
    interaction_activities_num, total_activities_num = 0, 0
    GID_value, GIP_value = 0, 0
    nodes_num = len(community_node)
    
    if nodes_num > 1:
        for (u, v), edges in PI.items():
            if u in community_node and v in community_node:
                total_activities_num += len(edges)
                if u != v:
                    interaction_activities_num += len(edges)

        if total_activities_num > 0:
            GIP_value = interaction_activities_num / total_activities_num
        
        if interaction_activities_num > 0:
            GID_value = interaction_activities_num / (nodes_num * (nodes_num - 1) * t_obs)

    return GIP_value, GID_value


def calc_cs_index(index, community_node, t_obs, method, rate, weights, LB_values, UB_values):

    EI_list, SIT_list, CED_list = [], [], []

    for u in tqdm.tqdm(community_node, desc="Calculating ATG-S measures for each user"):
        EI_u, SIT_u, CED_u = ATGS(index, u, community_node, t_obs, method, rate)
        EI_list.append(normalize_scores(EI_u, "EI", LB_values, UB_values))
        SIT_list.append(normalize_scores(SIT_u, "SIT", LB_values, UB_values))
        CED_list.append(normalize_scores(CED_u, "CED", LB_values, UB_values))
    
    EI_avg = round(np.mean(EI_list), 4)
    SIT_avg = round(np.mean(SIT_list), 4)
    CED_avg = round(np.mean(CED_list), 4)

    GIP, GID = GIS(index["PI"], community_node, t_obs)
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
    node_mapping_path = "./Datasets/Node_Mapping/"
    last_timestamps = {"BTW": 1506315747, "CC": 1643673425, "C26": 1672531185, "C144": 1672531150}

    for dataset in dataset_list:
        node_mapping = read_node_mapping(node_mapping_path + dataset + "_node_mapping.txt")

        dataset_path = data_path + dataset + "_attributed.txt"
        pro_dataset, t_obs = preprocess_dataset(dataset_path, dataset, node_mapping, last_timestamps[dataset])
        index, last_mutual_key, last_key =pp_i.buildPANEIndex(pro_dataset, node_mapping)
        LB_values, UB_values = pp_i.findBoundsPANE(index, t_obs, decay_method, decay_rate)
    
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
                community = [node_mapping[int(u)] for u in community]
                S, MS = calc_cs_index(index, community, t_obs, decay_method, decay_rate, measure_weights, LB_values, UB_values)
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
        output_file = community_dir + dataset + "_communities_cohesiveness_I_COHESION_w.txt"

        # Write the results to a txt file
        with open(output_file, "w") as f:
            f.write("Community\tMeasure Scores\tCohesiveness Score\tExplanation\tCompTime\tExpTime\n")
            for line in results:
                f.write(line)
        
        print(f"Results saved to {output_file}")