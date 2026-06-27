"""
This script runs synthetic datasets to test the efficiency of our COHESION algorithms
"""

import random
import networkx as nx
import sys
import time
import tqdm
from pympler import asizeof

target_path = "./"
sys.path.append(target_path)
from COHESION.Utils import read_node_mapping, buildPANEIndex
from COHESION.Preprocessing import preprocessing_o as pp_o, preprocessing_index as pp_i, preprocessing_index_rule1 as pp_ir1, preprocessing_index_rule2 as pp_ir2, preprocessing_index_trim as pp_it


# Define global variables for the start time of each dataset
DATASET_START_TIMES = {
    "AskUbuntu": 1231431607,
    "MathOverflow": 1254192988,
    "StackOverflow": 1217567877,
    "SuperUser": 1217651565
}


def graph_stats(attributed_G): 
    num_nodes = attributed_G.number_of_nodes()
    num_edges = attributed_G.number_of_edges()
    num_timestamps = len(set([int(d['timestamp']) for u, v, d in attributed_G.edges(data=True)]))
    density = nx.density(attributed_G)
    average_degree = sum(dict(attributed_G.degree()).values()) / num_nodes
    print(f"{num_nodes:,} & {num_edges:,} & {num_timestamps:,} & {density:.4f} & {average_degree:.2f}\\\\")


def graph_construction(attribute_file, dataset, t_obs):
    attributed_G = nx.MultiDiGraph()
    delta = t_obs - DATASET_START_TIMES[dataset]

    with open(attribute_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            u, v, timestamp, sentiment = parts[0], parts[1], int(parts[2]), int(parts[3])
            timestamp = timestamp - DATASET_START_TIMES[dataset]
        if timestamp <= delta:
            attributed_G.add_edge(u, v, timestamp=timestamp, sentiment=sentiment)
    return attributed_G, delta


def preprocess_dataset(file_path, dataset, t_obs):
    preprocessed_edges = []
    start_t = DATASET_START_TIMES[dataset]
    delta = t_obs - start_t
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            u_raw, v_raw, timestamp_delta, sentiment = parts[0], parts[1], int(parts[2]) - start_t, int(parts[3])
            if timestamp_delta <= delta:
                preprocessed_edges.append((int(u_raw), int(v_raw), timestamp_delta, sentiment))
    
    return preprocessed_edges, delta


if __name__ == "__main__":
    # Input parameters
    decay_method = "exp"
    decay_rate = 0.0001
    threshold_T = pow(10, -10)

    node_mapping_path = "./Datasets/Node_Mapping/"
    S_dataset_dir = "./Datasets/OSNs/"
    L_dataset_dir = "D:/TemporalNetworks/"

    S_dataset_list = ["BTW", "CC", "C26", "C144"]
    L_dataset_list = ["AskUbuntu", "StackOverflow", "MathOverflow", "SuperUser"]
    last_timestamps = {"BTW": 1506315747, "CC": 1643673425, "C26": 1672531185, "C144": 1672531150, "AskUbuntu": 1457266069, "StackOverflow": 1457273428, "MathOverflow": 1457262355, "SuperUser": 1457266493}

    result_timespent = {}
 

    for dataset in S_dataset_list + L_dataset_list:
        print(f"Processing dataset: {dataset}")

        # Step 1: Load the dataset
        if dataset in S_dataset_list:
            dataset_dir = S_dataset_dir
            attribute_file = dataset_dir + dataset + "_attributed.txt"
        else:
            dataset_dir = L_dataset_dir
            attribute_file = dataset_dir + dataset + "_sentiment.txt"
 

        # Step 2: Run preprocessing methods and record time taken
        timespent_list = []

        # Method 1: Index-oblivious method
        starttime = time.time()
        if dataset in S_dataset_list:
            attributed_G, t_obs = pp_o.graph_construction(dataset_dir + dataset + "_attributed.txt", dataset, last_timestamps[dataset])
        else:
            attributed_G, t_obs = graph_construction(attribute_file, dataset, last_timestamps[dataset])
        LB_values, UB_values = pp_o.findBounds(attributed_G, t_obs, decay_method, decay_rate)
        print("Minimum values:", LB_values)
        print("Maximum values:", UB_values)
        timespent_list.append(time.time() - starttime)

        # Method 2: Index-based method
        starttime = time.time()
        if dataset in S_dataset_list:
            node_mapping = read_node_mapping(node_mapping_path + dataset + "_node_mapping.txt")
            pro_dataset, t_obs = pp_i.preprocess_dataset(attribute_file, dataset, node_mapping, last_timestamps[dataset])
        else:
            pro_dataset, t_obs = preprocess_dataset(attribute_file, dataset, last_timestamps[dataset])
        
        preprocessTime = time.time() - starttime
        
        starttime = time.time()
        index, last_mutual_key, last_key = buildPANEIndex(pro_dataset)
        indexConstructionTime = time.time() - starttime
        print(f"Size of the PANE-Index: {asizeof.asizeof(index) / 1024**2:.2f} MB")

        starttime = time.time()
        LB_values, UB_values = pp_i.findBoundsPANE(index, t_obs, decay_method, decay_rate)
        print("Minimum values:", LB_values)
        print("Maximum values:", UB_values)
        timespent_list.append(time.time() - starttime + preprocessTime + indexConstructionTime)

        # Method 3: Index-based method with pruning rule 1
        starttime = time.time()   
        start_t = pp_ir1.findStartTime(t_obs, decay_rate, threshold_T)
        trimmed_index, mutual_nodes = pp_ir1.trimPANEIndex(index, start_t)
        print(f"Size of the trimmed PANE-Index: {asizeof.asizeof(trimmed_index) / 1024**2:.2f} MB")

        LB_values, UB_values = pp_ir1.findBoundsPANE(trimmed_index, t_obs, decay_method, decay_rate)
        print("Minimum values:", LB_values)
        print("Maximum values:", UB_values)
        timespent_list.append(time.time() - starttime + preprocessTime + indexConstructionTime)

        # Method 4: Index-based method with pruning rule 2
        starttime = time.time()
        LB_values, UB_values = pp_ir2.findBoundsPANE(index, t_obs, last_mutual_key, last_key, decay_method, decay_rate)
        print("Minimum values:", LB_values)
        print("Maximum values:", UB_values)
        timespent_list.append(time.time() - starttime + preprocessTime + indexConstructionTime)

        # Method 5: Index-based method with both pruning
        starttime = time.time()
        start_t = pp_it.findStartTime(t_obs, decay_rate, threshold_T)
        trimmed_index, mutual_nodes = pp_it.trimPANEIndex(index, start_t)
        
        LB_values, UB_values = pp_it.findBoundsPANE(trimmed_index, mutual_nodes, t_obs, last_mutual_key, last_key, decay_method, decay_rate)
        print("Minimum values:", LB_values)
        print("Maximum values:", UB_values)
        timespent_list.append(time.time() - starttime + preprocessTime + indexConstructionTime)

        result_timespent[dataset] = timespent_list

        graph_stats(attributed_G)
        print("\n")

    # Print out the results
    print("\nTime taken for each method on each dataset:")
    for dataset, timespent_list in result_timespent.items():
        print(f"\n{dataset}:")
        for i, time_taken in enumerate(timespent_list):
            print(f"  Method {i+1}: {time_taken:.4f} seconds")