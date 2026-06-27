"""
This script is used to find the boundaries of each measure given a graph (basic method)

Input: Graph, decay_method, decay_rate
Output: min-max of five measures
"""
from tqdm import tqdm
import sys
import numpy as np
from pympler import asizeof


target_path = "./"
sys.path.append(target_path)
from COHESION.Utils import buildPANEIndex, trimPANEIndex, calcEnjoyment, getEdges, getPairEdges, preprocess_dataset, time_call, read_node_mapping, getATGSBounds, getGIDBounds



def getMutualNodes(NI):
    mutual_nodes = set()
    for node, (_, mutual_neighbors) in NI.items():
        if mutual_neighbors:
            mutual_nodes.add(node)
            for neighbor in mutual_neighbors:
                mutual_nodes.add(neighbor)
    return mutual_nodes


# Calculate the threshold for EI
def calcThresholdEI(PI, NI, node_set, key, t_obs, method, rate):
    # Find the pair with last timestamp
    u, v = key
    uE_EI, vE_EI = -float('inf'), -float('inf')

    # get edge sets for both node in the pair
    uE_pos, _ = getEdges(PI, u, list(NI[u][0]) + list(NI[u][1]), trim=False)
    uE_EI = calcEnjoyment(uE_pos, t_obs, method, rate)
    node_set.remove(u)
    
    if u == v:
        return node_set, uE_EI
 
    vE_pos, _ = getEdges(PI, v, list(NI[v][0]) + list(NI[v][1]), trim=False)
    vE_EI = calcEnjoyment(vE_pos, t_obs, method, rate)
    node_set.remove(v)

    return node_set, max(uE_EI, vE_EI)


# Calculate the threshold for SIT
def calcThresholdSIT(PI, NI, node_mutual_set, key, t_obs, method, rate):
    # Find the pair with last mutual timestamp
    if not node_mutual_set:
        return node_mutual_set, 0
    
    u, v = key
    uME_SIT, vME_SIT = -float('inf'), -float('inf')

    # get mutual edge sets for both node in the pair
    uME_pos = getPairEdges(PI, u, list(NI[u][1]), trim=False, sort=True)
    vME_pos = getPairEdges(PI, v, list(NI[v][1]), trim=False, sort=True)

    uME_SIT, vME_SIT = 0, 0
    for _, edge_list in uME_pos.items():
        val = calcEnjoyment(edge_list, t_obs, method, rate)
        uME_SIT += val if not np.isnan(val) else 0
    for _, edge_list in vME_pos.items():
        val = calcEnjoyment(edge_list, t_obs, method, rate)
        vME_SIT += val if not np.isnan(val) else 0

    node_mutual_set.remove(u)
    node_mutual_set.remove(v)

    return node_mutual_set, max(uME_SIT, vME_SIT)


@time_call("finding the bounds of each measure")
def findBoundsPANE(index, t_obs, last_mutual_key, last_key, method, rate):
    psyM = ['EI', 'SIT', 'CED', 'GIP', 'GID']
    minValues = {m: 0 for m in psyM}
    maxValues = {m: 0 for m in psyM}
    node_set = list(index["NI"].keys())
    
    # Take extreme positive configuration as an example
    PI_pos = {key: [(t, 1) for (t, _) in index["PI"][key]] for key in index["PI"]}
    NI = index["NI"]
    mutual_nodes = getMutualNodes(NI)

    # Calculate the thresholds based on the last interactions
    node_set, EI_threshold = calcThresholdEI(PI_pos, NI, node_set, last_key, t_obs, method, rate)
    mutual_nodes, SIT_threshold = calcThresholdSIT(PI_pos, NI, mutual_nodes, last_mutual_key, t_obs, method, rate)
    maxValues['EI'] = max(maxValues['EI'], EI_threshold)
    maxValues['SIT'] = max(maxValues['SIT'], SIT_threshold)
   
    # Separate the boundary calculation for EI and SIT to optimize performance
    for u in tqdm(node_set, desc="Calculating EI boundaries for each user"):
        uE_pos, uE_count = getEdges(PI_pos, u, NI[u][0] | NI[u][1], trim=False)
        max_edge_EI = calcEnjoyment([uE_pos[-1]], t_obs, method, rate) if uE_pos else 0
    
        if max_edge_EI * uE_count >= EI_threshold:  # A quick check to skip unnecessary EI calculations
            EI_value = calcEnjoyment(uE_pos, t_obs, method, rate)
            maxValues['EI'] = max(maxValues['EI'], EI_value)

    for u in tqdm(mutual_nodes, desc="Calculating SIT boundaries for each user"):
        uME_pos = getPairEdges(PI_pos, u, list(NI[u][1]), trim=False, sort=True)
        max_edge_SIT = calcEnjoyment(list(uME_pos.values())[0], t_obs, method, rate) if uME_pos else 0

        if max_edge_SIT * len(uME_pos) >= SIT_threshold:  # A quick check to skip unnecessary SIT calculations       
            SIT_value = 0
            for _, edges in uME_pos.items():
                val = calcEnjoyment(edges, t_obs, method, rate)
                SIT_value += val if not np.isnan(val) else 0
            maxValues['SIT'] = max(maxValues['SIT'], SIT_value)

    minValues, maxValues = getATGSBounds(minValues, maxValues)
    minValues["GIP"], maxValues["GIP"] = 0, 1 # Property 3
    minValues, maxValues = getGIDBounds(PI_pos, t_obs, minValues, maxValues, trim=False)

    return minValues, maxValues


if __name__ == "__main__":
    dataset_dir = "./Datasets/OSNs/"
    node_mapping_file = "./Datasets/Node_Mapping/C144_node_mapping.txt"
    dataset = "C144"

    # Input parameters
    decay_method = "exp"
    decay_rate = 0.0001
    t_obs = 1672531150
    threshold_T = pow(10, -10)

    # Preprocess the dataset
    node_mapping = read_node_mapping(node_mapping_file)
    dataset_path = dataset_dir + dataset + "_attributed.txt"
    pro_dataset, t_obs = preprocess_dataset(dataset_path, dataset, node_mapping, t_obs)

    # Based on the dataset, construct an indexed structure
    index, last_mutual_key, last_key = buildPANEIndex(pro_dataset)
    print(f"Size of the PANE-Index: {asizeof.asizeof(index) / 1024**2:.2f} MB")

    # Calculate the min-max values for each measure
    LB_values, UB_values = findBoundsPANE(index, t_obs, last_mutual_key, last_key, decay_method, decay_rate)
    print("Minimum values:", LB_values)
    print("Maximum values:", UB_values)

    
    # Store the min-max values in a txt file
    # output_file = dataset_dir + dataset.split('_attributed.txt')[0] + f"_bounds.txt"
    # with open(output_file, 'w') as f:
    #     f.write("Measure\tMin\tMax\n")
    #     for measure in LB_values.keys():
    #         f.write(f"{measure}\t{LB_values[measure]}\t{UB_values[measure]}\n")
