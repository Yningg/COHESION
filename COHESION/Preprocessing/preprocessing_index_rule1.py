"""
This script is used to find the boundaries of each measure given a graph (basic method)

Input: Graph, decay_method, decay_rate
Output: min-max of five measures
"""
from tqdm import tqdm
import sys
from pympler import asizeof


target_path = "./"
sys.path.append(target_path)
from COHESION.Utils import buildPANEIndex, trimPANEIndex, calcEnjoyment, findStartTime, getEdges, getPairEdges, preprocess_dataset, time_call, read_node_mapping, getATGSBounds, getGIDBounds


def getNodesSet(PI):
    node_set = set()
    for (u, v), info, in PI.items():
        if info[1]:  # only consider pairs with edges
            node_set.add(u)
            node_set.add(v)
    return node_set


# Calculate the threshold for EI
def calcThresholdEI(PI, NI, node_set, key, t_obs, method, rate):
    # Find the pair with last timestamp
    u, v = key
    uE_EI, vE_EI = -float('inf'), -float('inf')

    # get edge sets for both node in the pair
    uE_pos, _ = getEdges(PI, u, list(NI[u][0]) + list(NI[u][1]), trim=True)
    uE_EI = calcEnjoyment(uE_pos, t_obs, method, rate)
    node_set.remove(u)
    
    if u == v:
        return node_set, uE_EI
 
    vE_pos, _ = getEdges(PI, v, list(NI[v][0]) + list(NI[v][1]), trim=True)
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
    uME_pos = getPairEdges(PI, u, list(NI[u][1]), trim=True, sort=False)
    vME_pos = getPairEdges(PI, v, list(NI[v][1]), trim=True, sort=False)

    uME_SIT, vME_SIT = 0, 0
    for _, edge_list in uME_pos.items():
        uME_SIT += calcEnjoyment(edge_list, t_obs, method, rate)
    for _, edge_list in vME_pos.items():
        vME_SIT += calcEnjoyment(edge_list, t_obs, method, rate)
    
    node_mutual_set.remove(u)
    node_mutual_set.remove(v)

    return node_mutual_set, max(uME_SIT, vME_SIT)


@time_call("finding the bounds of each measure")
def findBoundsPANE(trimmed_index, t_obs, method, rate):
    psyM = ['EI', 'SIT', 'CED', 'GIP', 'GID']
    minValues = {m: 0 for m in psyM}
    maxValues = {m: 0 for m in psyM}
    
    # Take extreme positive configuration as an example
    trimmed_PI_pos = {key: (trimmed_index["PI"][key][0], [(t, 1) for (t, _) in trimmed_index["PI"][key][1]]) for key in trimmed_index["PI"]}
    trimmed_NI = trimmed_index["NI"]
    node_set = getNodesSet(trimmed_PI_pos)
    
    for u in tqdm(node_set, desc="Calculating boundaries for each user"):
        uE_pos, _ = getEdges(trimmed_PI_pos, u, list(trimmed_NI[u][0]) + list(trimmed_NI[u][1]), trim=True)
        uME_pos = getPairEdges(trimmed_PI_pos, u, list(trimmed_NI[u][1]), trim=True, sort=False)

        # Calculate the Enjoyment Index (EI) of node u in subgraph H at time t_obs
        EI_value = calcEnjoyment(uE_pos, t_obs, method, rate)
        maxValues['EI'] = max(maxValues['EI'], EI_value)

        SIT_value = 0
        for pair, edges in uME_pos.items():
            SIT_value += calcEnjoyment(edges, t_obs, method, rate)
            if u == '337631':
                print(f"User {pair}: SIT value = {calcEnjoyment(edges, t_obs, method, rate)}")
        maxValues['SIT'] = max(maxValues['SIT'], SIT_value)
     

    minValues, maxValues = getATGSBounds(minValues, maxValues)
    minValues["GIP"], maxValues["GIP"] = 0, 1 # Property 3
    minValues, maxValues = getGIDBounds(trimmed_index["PI"], t_obs, minValues, maxValues, trim=True)

    return minValues, maxValues


if __name__ == "__main__":
    # Input parameters
    decay_method = "exp"
    decay_rate = 0.0001
    t_obs = 1672531150
    threshold_T = pow(10, -10)

    # Preprocess the dataset
    node_mapping_file = "./Datasets/Node_Mapping/C144_node_mapping.txt"
    node_mapping = read_node_mapping(node_mapping_file)
    pro_dataset, t_obs = preprocess_dataset("./Datasets/OSNs/C144_attributed.txt", "C144", node_mapping, t_obs)

    # Based on the dataset, construct an indexed structure
    start_t = findStartTime(t_obs, decay_rate, threshold_T)
    index, last_mutual_key, last_key = buildPANEIndex(pro_dataset)
    print(f"Size of the PANE-Index: {asizeof.asizeof(index) / 1024**2:.2f} MB")

    # Trim the index to save space
    trimmed_index, mutual_nodes = trimPANEIndex(index, start_t)
    print(f"Size of the trimmed PANE-Index: {asizeof.asizeof(trimmed_index) / 1024**2:.2f} MB")

    # Calculate the min-max values for each measure
    LB_values, UB_values = findBoundsPANE(trimmed_index, t_obs, decay_method, decay_rate)
    print("Minimum values:", LB_values)
    print("Maximum values:", UB_values)

    
    # Store the min-max values in a txt file
    # output_file = dataset_dir + dataset.split('_attributed.txt')[0] + f"_bounds.txt"
    # with open(output_file, 'w') as f:
    #     f.write("Measure\tMin\tMax\n")
    #     for measure in LB_values.keys():
    #         f.write(f"{measure}\t{LB_values[measure]}\t{UB_values[measure]}\n")
