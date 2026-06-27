"""
This script runs generated dense datasets to test the efficiency of our COHESION algorithms
"""
import networkx as nx
import sys
import time
import matplotlib.pyplot as plt
from pympler import asizeof

target_path = "./"
sys.path.append(target_path)
from COHESION.Preprocessing import preprocessing_o as pp_o, preprocessing_index as pp_i, preprocessing_index_rule1 as pp_ir1, preprocessing_index_rule2 as pp_ir2, preprocessing_index_trim as pp_it


def graph_construction(attributed_file, t_obs, t_start):
    with open(attributed_file, 'r') as f:
        edge_list = []
        for line in f:
            u, v, t, s = line.strip().split("\t")
            edge_list.append((int(u), int(v), int(t), int(s)))

    attributed_G = nx.MultiDiGraph()
    delta = t_obs - t_start

    for u, v, timestamp, sentiment in edge_list:
        timestamp = timestamp - t_start
        if timestamp <= delta:
            attributed_G.add_edge(u, v, timestamp=timestamp, sentiment=sentiment)
    return attributed_G, delta


def preprocess_dataset(attribute_file, t_obs, t_start):
    preprocessed_edges = []
    delta = t_obs - t_start
    
    with open(attribute_file, 'r') as f:
        for line in f:
            u, v, timestamp, sentiment = line.strip().split("\t")
            timestamp_delta = int(timestamp) - t_start
            if timestamp_delta <= delta:
                preprocessed_edges.append((int(u), int(v), timestamp_delta, int(sentiment)))

    return preprocessed_edges, delta


if __name__ == '__main__':
    # Input parameters
    decay_method = "exp"
    decay_rate = 0.0001
    threshold_T = pow(10, -10)

    node_mapping_path = "./Datasets/Node_Mapping/"
    dataset_dir = "./Datasets/SyntheticOSNs/"
    dataset_list = [f"Dense_Multidigraph_{i/10}" for i in range(1, 10)]
    dataset_start_times = [6, 304, 38, 48, 64, 296, 74, 99, 8]
    last_timestamps = [15845381, 32732725, 48513060, 65474688, 81175538, 101792241, 115480231, 131526894, 149512435]
    result_timespent = {}

    """
    For each dataset, we run the following steps:
    1. Run five preprocessing methods and record the time taken for each method
    2. Print out number of nodes, number of edges, time span, density, and average degree
    """

    for i, dataset in enumerate(dataset_list):
        print(f"Processing dataset: {dataset}")
        attribute_file = dataset_dir + dataset + ".txt"
        timespent_list = []

        # Method 1: Index-oblivious method
        starttime = time.time()
        attributed_G, t_obs = graph_construction(attribute_file, last_timestamps[i], dataset_start_times[i])
        LB_values, UB_values = pp_o.findBounds(attributed_G, t_obs, decay_method, decay_rate)
        print("Minimum values:", LB_values)
        print("Maximum values:", UB_values)
        timespent_list.append(time.time() - starttime)

        # Method 2: Index-based method
        starttime = time.time()
        pro_dataset, t_obs = preprocess_dataset(attribute_file, last_timestamps[i], dataset_start_times[i])
        index, last_mutual_key, last_key = pp_i.buildPANEIndex(pro_dataset)
        LB_values, UB_values = pp_i.findBoundsPANE(index, t_obs, decay_method, decay_rate)
        print("Minimum values:", LB_values)
        print("Maximum values:", UB_values)
        timespent_list.append(time.time() - starttime)

        # Method 3: Index-based method with pruning rule 1
        starttime = time.time()
        pro_dataset, t_obs = preprocess_dataset(attribute_file, last_timestamps[i], dataset_start_times[i])
        index, last_mutual_key, last_key = pp_ir1.buildPANEIndex(pro_dataset)    
        start_t = pp_ir1.findStartTime(t_obs, decay_rate, threshold_T)
        trimmed_index, mutual_nodes = pp_ir1.trimPANEIndex(index, start_t)
        LB_values, UB_values = pp_ir1.findBoundsPANE(trimmed_index, t_obs, decay_method, decay_rate)
        print("Minimum values:", LB_values)
        print("Maximum values:", UB_values)
        timespent_list.append(time.time() - starttime)

        # Method 4: Index-based method with pruning rule 2
        starttime = time.time()
        pro_dataset, t_obs = preprocess_dataset(attribute_file, last_timestamps[i], dataset_start_times[i])
        index, last_mutual_key, last_key = pp_ir2.buildPANEIndex(pro_dataset)

        LB_values, UB_values = pp_ir2.findBoundsPANE(index, t_obs, last_mutual_key, last_key, decay_method, decay_rate)
        print("Minimum values:", LB_values)
        print("Maximum values:", UB_values)
        timespent_list.append(time.time() - starttime)

        # Method 5: Index-based method with both pruning rules
        starttime = time.time()
        pro_dataset, t_obs = preprocess_dataset(attribute_file, last_timestamps[i], dataset_start_times[i])
        index, last_mutual_key, last_key = pp_it.buildPANEIndex(pro_dataset)
        start_t = pp_it.findStartTime(t_obs, decay_rate, threshold_T)
        trimmed_index, mutual_nodes = pp_it.trimPANEIndex(index, start_t)
  
        LB_values, UB_values = pp_it.findBoundsPANE(trimmed_index, mutual_nodes, t_obs, last_mutual_key, last_key, decay_method, decay_rate)
        print("Minimum values:", LB_values)
        print("Maximum values:", UB_values)
        timespent_list.append(time.time() - starttime)

        result_timespent[dataset] = timespent_list

        print("\n")

    # Print out the results
    print("\nTime taken for each method on each dataset:")
    for dataset, timespent_list in result_timespent.items():
        print(f"\n{dataset}:")
        for i, time_taken in enumerate(timespent_list):
            print(f"  Method {i+1}: {time_taken:.4f} seconds")

    save_path = "./Figures/Efficiency_Visualization_Results/"
    # Plotting the results as a line chart, x: density, y: time taken, different lines for different methods
    method_list = ["IO-PP", "I-PP-w", "I-PP-R1", "I-PP-R2", "I-PP"]
    marker_list = ['o', 's', 'D', '^', 'v']
    color_list = ['#a5d1f2', "#d1d1d1","#8ab3acff", "#529cb4", "#595959"]
    font_size = 16

    fig, ax = plt.subplots(figsize=(8, 3.5))
    for i in range(5):
        method_times = [result_timespent[dataset][i] for dataset in dataset_list]
        ax.plot(dataset_list, method_times, marker=marker_list[i], color=color_list[i], label=method_list[i], linewidth=2.5, markersize=9)

    ax.set_xticks(dataset_list)
    ax.set_xticklabels([f"{x / 10:.1f}" for x in range(1, 10)], fontsize=font_size)

    ax.set_xlabel("Density", fontsize=font_size)
    ax.set_ylabel("Runtime (s)", fontsize=font_size)

    ax.tick_params(axis="y", labelsize=font_size)
    ax.tick_params(axis="x", labelsize=font_size)

    fig.tight_layout()
    plt.subplots_adjust(left=0.12, right=0.95, top=0.85, bottom=0.2, wspace=0.09, hspace=0.25)
    plt.legend(loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.23), fontsize=font_size, frameon=True, columnspacing=0.35, handletextpad=0.25)
    plt.savefig(save_path + "DenseGraph_Efficiency.png", dpi=300, bbox_inches='tight')
    plt.show()