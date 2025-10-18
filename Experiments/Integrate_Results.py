"""
Extract and integrate communities identified by different algorithms for each dataset.
"""

import os
import ast
import numpy as np


def read_node_mapping(node_mapping_file):
    node_mapping = {}
    with open(node_mapping_file, 'r') as f:
        lines = f.readlines()
        # Reverse the mapping to map the new nodes back to the original nodes
        node_mapping = {int(line.split("\t")[1]): int(line.split("\t")[0]) for line in lines}
    
    return node_mapping


def group_results(results):
    grouped_results = {}
    for result in results:
        node = result[0]
        if node in grouped_results:
            grouped_results[node].append(result)
        else:
            grouped_results[node] = [result]
    return grouped_results


def get_condensed_results(grouped_results, dim_index):
    condensed_results = {}
    for node, result_list in grouped_results.items():
        cohesiveness_score = []
        valid_count = 0

        for result in result_list:
            if len(result[dim_index]) > 0 and result[dim_index][0] != 'Invalid':
                cohesiveness_score.append(result[dim_index])
                valid_count += 1
        
        if valid_count == 0:
            cohesiveness_score = ['Invalid'] * 5
        else:
            cohesiveness_avg = np.mean(cohesiveness_score, axis=0)
            cohesiveness_std = np.std(cohesiveness_score, axis=0)
            condensed_results[node] = {"avg": list(cohesiveness_avg), "std": list(cohesiveness_std)}

    return condensed_results

    

def process_algo_results(algo_result_dir, algorithm, node_mapping, file):
    results = []
    with open(algo_result_dir + file, 'r') as f:
        lines = f.readlines()
        if algorithm in ["ALS", "WCF-CRC", "I2ACSM"]:
            for line in lines:
                parts = line.strip().split("\t")
                community_node_list = list(ast.literal_eval(parts[3]))
                community_node_list = [str(node) for node in community_node_list]
                results.append(community_node_list)
        
        elif algorithm in ["ST-Exa", "CSD", "Repeeling"]:
            for line in lines:
                parts = line.strip().split("\t")
                community_node_list = ast.literal_eval(parts[2])
                community_node_list = [str(node_mapping[node]) for node in community_node_list]
                results.append(community_node_list)

        elif algorithm in ["TransZero_LS"]:
            for line in lines:
                parts = line.strip().split("\t")
                community_node_list = ast.literal_eval(parts[1].strip())
                community_node_list = [str(node_mapping[node]) for node in community_node_list]
                results.append(community_node_list)
    return results


def process_dataset(algorithm, dataset_list):
    algo_result_dir = output_dir + algorithm + "_Results/"
    files = [file for file in os.listdir(algo_result_dir) if any(name in file for name in dataset_list)]

    # For four files in the directory
    for file in files:
        print(f"Processing {file}...")

        # Read the results
        dataset_name = [name for name in dataset_list if name in file][0]
        node_mapping_file = node_mapping_dir + dataset_name + "_node_mapping.txt"
        node_mapping = read_node_mapping(node_mapping_file)

        results = process_algo_results(algo_result_dir, algorithm, node_mapping, file)
        results = [res for res in results if len(res) > 0]  # Filter out empty results
        results = [list(s) for s in set(frozenset(inner_list) for inner_list in results)] # Remove duplicate communities

        print(f"Number of non-empty results without duplication: {len(results)}")

        # Save the results in a single txt file
        output_file = output_dir + dataset_name + "/" + file.split(".txt")[0] + "_community.txt"
        with open(output_file, 'w') as f:
            for community in results:
                f.write(f"{community}\n")


# Output directory for the condensed results
node_mapping_dir = "./Datasets/Node_Mapping/"
output_dir = "./Datasets/Communities/"

dataset_list = ["BTW", "CC", "C26", "C144"]
dataset_sublist = ["BTW", "CC"]

algorithm_dict = {
    "ALS": {"dataset_list": dataset_list},
    "WCF-CRC": {"dataset_list": dataset_list},
    "CSD": {"dataset_list": dataset_list},
    "ST-Exa": {"dataset_list": dataset_list},
    "Repeeling": {"dataset_list": dataset_sublist},
    "I2ACSM": {"dataset_list": dataset_list},
    "TransZero_LS": {"dataset_list": dataset_list}
}

# Deal with results
for algorithm, content in algorithm_dict.items():
    process_dataset(algorithm, content["dataset_list"])


# Integrate communities
for dataset in dataset_list:
    print(f"Integrating results for {dataset}...")

    integrated_results = []
    files = [file for file in os.listdir(output_dir + dataset + "/") if "community.txt" in file]
    for input_file in files:
        with open(output_dir + dataset + '/' + input_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                community_node_list = ast.literal_eval(line.strip())
                integrated_results.append(community_node_list)
    
    # Remove duplicate communities
    integrated_results = [list(s) for s in set(frozenset(inner_list) for inner_list in integrated_results)]
    print(f"Number of non-empty results without duplication: {len(integrated_results)}")

    # Only keep communities with size >=3
    integrated_results = [community for community in integrated_results if len(community) >= 3]
    print(f"Number of communities with size >=3: {len(integrated_results)}")

    # Order the communities by size (smaller to larger)
    integrated_results = sorted(integrated_results, key=lambda x: len(x))

    # Save the integrated results
    output_file = output_dir + dataset + "/" + "Integrated_Results_" + dataset + "_community.txt"
    with open(output_file, 'w') as f:
        for community in integrated_results:
            f.write(f"{community}\n")
