"""
For communities in each dataset, divide them into 5 groups based on their sizes.
"""
import os
import numpy as np
import ast


def read_communities(file_path):
    communities = []
    with open(file_path, 'r') as f:
        for line in f:
            community = ast.literal_eval(line.strip())
            communities.append(community)
    return communities


def divide_communities_by_size(node_counts, num_groups=5):
    min_size = min(node_counts)
    max_size = max(node_counts)
    
    bins = np.logspace(np.log10(min_size), np.log10(max_size), num_groups + 1)
    bins = np.ceil(bins).astype(int)

    ranges = []
    ranges_details = []
    for i in range(len(bins) - 1):
        lower_bound = bins[i]
        upper_bound = bins[i + 1] - 1 if i < len(bins) - 2 else bins[i + 1]
        ranges.append((lower_bound, upper_bound))

        counts_in_range = [count for count in node_counts if lower_bound <= count <= upper_bound]
        ranges_details.append({"lower_bound": lower_bound, "upper_bound": upper_bound,
                               "max_in_range": max(counts_in_range) if counts_in_range else None,
                               "min_in_range": min(counts_in_range) if counts_in_range else None,
                               "num_communities": len(counts_in_range)})
    return ranges, ranges_details



if __name__ == "__main__":
    dir = "./Datasets/Communities/"
    datasets = ["BTW", "CC", "C26", "C144"]

    for dataset in datasets:
        file_name = dir + dataset + "/" + "Integrated_Results_" + dataset + "_community.txt"
        communities = read_communities(file_name)
        print(f"Dataset: {dataset}, Total communities: {len(communities)}")
        node_counts = [len(comm) for comm in communities]
        print(f"Min size: {min(node_counts)}, Max size: {max(node_counts)}")
        
        ranges, ranges_details = divide_communities_by_size(node_counts, 5)
        
        for i, detail in enumerate(ranges_details):
            print(f"Group {i+1}: Size range {detail['lower_bound']}-{detail['upper_bound']}, "
                  f"Min: {detail['min_in_range']}, Max: {detail['max_in_range']}, "
                  f"Num communities: {detail['num_communities']}")