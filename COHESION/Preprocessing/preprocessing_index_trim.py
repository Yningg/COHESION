"""
This script is used to find the boundaries of each measure given a graph (basic method)

Input: Graph, decay_method, decay_rate
Output: min-max of five measures
"""
from collections import defaultdict
import bisect
from tqdm import tqdm
import sys
from pympler import asizeof


target_path = "./"
sys.path.append(target_path)
from COHESION.Utils import ESenti, time_decay, calcEnjoyment, findStartTime, getEdges, getPairEdges, preprocess_dataset, time_call, read_node_mapping


# Build the PANE-Index for the graph
def buildPANEIndex(dataset, node_mapping):
    PI = defaultdict(list) # for each pair, store edge list
    pair_masks = defaultdict(int)   # for each pair, store direction bitmask
    NI = defaultdict(lambda: [set(), set()]) # for each node, store directed neighbors and mutual neighbors
    last_key = ()
    last_mutual_t, last_mutual_key = -1, ()

    for u_raw, v_raw, timestamp, sentiment in dataset:
        u, v = node_mapping[int(u_raw)], node_mapping[int(v_raw)]
        if u < v:
            key = (u, v)
            direction_bit = 1 # u -> v
        else:
            key = (v, u)
            direction_bit = 2 # v -> u
        
        if key not in PI:
            PI[key] = []
            pair_masks[key] = 0
        
        # Update the edge list for the existing pair and bitmask
        PI[key].append((timestamp, sentiment))
        pair_masks[key] |= direction_bit
    
        # Since the dataset is already sorted, the last_t is the timestamp of the last edge
        last_key = key

    
    for (u, v), edges in PI.items():
        mask = pair_masks[(u, v)]
        
        is_mutual = (mask == 3)  # Check Mutual (3 = both directions set)
        target_type = 1 if is_mutual else 0

        if u == v:
            NI[u][0].add(v) # self-loops are treated as directed
        else:
            NI[u][target_type].add(v)
            NI[v][target_type].add(u)

            if is_mutual and edges[-1][0] > last_mutual_t:
                last_mutual_t = edges[-1][0]
                last_mutual_key = (u, v)
        
    print(f"Last mutual interaction timestamp: {last_mutual_t}")
    print(f"Last overall interaction key: {last_key}")
    print(f"Last mutual interaction key: {last_mutual_key}")
    return {"PI": PI, "NI": NI}, last_mutual_key, last_key


def trimPANEIndex(index, start_t):
    trimmed_PI, trimmed_NI = {}, {}
    mutual_nodes = set()
    removed_pairs = set()

    for (u, v), edges in index["PI"].items():
        total_count = len(edges)
        
        valid_edges = []
        if edges[-1][0] < start_t:
            removed_pairs.add((u, v))
            pass 
        elif edges[0][0] >= start_t:
            valid_edges = edges
        else:
            idx = bisect.bisect_left([edge[0] for edge in edges], start_t)
            valid_edges = edges[idx:]

        trimmed_PI[(u, v)] = (total_count, valid_edges)

    for u, (dN, mN) in index["NI"].items():
        dN_new = {v for v in dN if (min(u, v), max(u, v)) not in removed_pairs}
        mN_new = {v for v in mN if (min(u, v), max(u, v)) not in removed_pairs}
        if dN_new or mN_new:
            trimmed_NI[u] = [dN_new, mN_new]
            if mN_new:
                mutual_nodes.add(u)

    return {"PI": trimmed_PI, "NI": trimmed_NI}, mutual_nodes


def getNodesSet(PI):
    node_set = set()
    for (u, v), info, in PI.items():
        if info[1]:  # only consider pairs with edges
            node_set.add(u)
            node_set.add(v)
    return node_set


# Calculate the threshold for EI
def calcThresholdEI(index, node_set, key, t_obs, method, rate):
    # Find the pair with last timestamp
    u, v = key
    uE_EI, vE_EI = -float('inf'), -float('inf')

    # get edge sets for both node in the pair
    uE_pos = getEdges(index["PI"], u, index["NI"][u][0] | index["NI"][u][1])
    uE_EI = calcEnjoyment(uE_pos, t_obs, rate, method)
    node_set.remove(u)
    
    if u == v:
        return node_set, uE_EI
 
    vE_pos = getEdges(index["PI"], v, index["NI"][v][0] | index["NI"][v][1])
    vE_EI = calcEnjoyment(vE_pos, t_obs, rate, method)
    node_set.remove(v)

    return node_set, max(uE_EI, vE_EI)


# Calculate the threshold for SIT
def calcThresholdSIT(index, node_mutual_set, key, t_obs, method, rate):
    # Find the pair with last mutual timestamp
    u, v = key
    uME_SIT, vME_SIT = -float('inf'), -float('inf')

    # get mutual edge sets for both node in the pair
    uME_pos = getPairEdges(index["PI"], u, index["NI"][u][1])
    vME_pos = getPairEdges(index["PI"], v, index["NI"][v][1])
    
    uME_SIT, vME_SIT = 0, 0
    for _, edge_list in uME_pos.items():
        uME_SIT += calcEnjoyment(edge_list, t_obs, rate, method)
    for _, edge_list in vME_pos.items():
        vME_SIT += calcEnjoyment(edge_list, t_obs, rate, method)
    
    node_mutual_set.remove(u)
    node_mutual_set.remove(v)

    return node_mutual_set, max(uME_SIT, vME_SIT)


# Get the bounds of ATG-S measures
def getATGSBounds(ATGS_values, minValues, maxValues):
    maxValues['EI'], maxValues['SIT'] = round(max(ATGS_values['EI']), 4), round(max(ATGS_values['SIT']), 4)
    maxValues['CED'] = maxValues['EI']
    minValues['EI'], minValues['SIT'], minValues['CED'] = -maxValues['EI'], -maxValues['SIT'], -maxValues['CED']
    return minValues, maxValues


# Get the bounds of GI-S measures
def getGIDBounds(PI_index, t_obs, minValues, maxValues):
    minValues["GID"] = 0
    interaction_list = [value[0] for key, value in PI_index.items() if key[0] != key[1]]
    maxValues["GID"] = round(max(interaction_list) / (2 * t_obs), 4) if t_obs > 0 else 1

    return minValues, maxValues


def findBoundsPANE(trimmed_index, mutual_nodes, t_obs, last_mutual_key, last_key, method, rate):
    psyM = ['EI', 'SIT', 'CED', 'GIP', 'GID']
    minValues = {m: 0 for m in psyM}
    maxValues = {m: 0 for m in psyM}
    ATGS_values = {'EI': [], 'SIT': []}
    
    # Take extreme positive configuration as an example
    trimmed_PI_pos = {key: (trimmed_index["PI"][key][0], [(t, 1) for (t, _) in trimmed_index["PI"][key][1]]) for key in trimmed_index["PI"]}
    trimmed_index_pos = {"PI": trimmed_PI_pos, "NI": trimmed_index["NI"]}
    node_set = getNodesSet(trimmed_PI_pos)

    # Calculate the thresholds based on the last interactions
    node_set, EI_threshold = calcThresholdEI(trimmed_index_pos, node_set, last_key, t_obs, method, rate)
    mutual_nodes, SIT_threshold = calcThresholdSIT(trimmed_index_pos, mutual_nodes, last_mutual_key, t_obs, method, rate)
    ATGS_values['EI'].append(EI_threshold)
    ATGS_values['SIT'].append(SIT_threshold)
   
    # Separate the boundary calculation for EI and SIT to optimize performance
    for u in tqdm(node_set, desc="Calculating EI boundaries for each user"):
        uE_pos = getEdges(trimmed_PI_pos, u, trimmed_index["NI"][u][0] | trimmed_index["NI"][u][1])
        max_edge_EI = ESenti(uE_pos[-1][0], uE_pos[-1][1], uE_pos[:-1], rate, method) * time_decay(t_obs, uE_pos[-1][0], rate, method)
    
        if max_edge_EI * len(uE_pos) >= EI_threshold:  # A quick check to skip unnecessary EI calculations
            ATGS_values['EI'].append(calcEnjoyment(uE_pos, t_obs, rate, method))
        
    for u in tqdm(mutual_nodes, desc="Calculating SIT boundaries for each user"):
        uME_pos = getPairEdges(trimmed_PI_pos, u, trimmed_index["NI"][u][1])
        max_edge_SIT = calcEnjoyment(list(uME_pos.values())[0], t_obs, rate, method)

        if max_edge_SIT * len(uME_pos) >= SIT_threshold:  # A quick check to skip unnecessary SIT calculations       
            SIT_value = 0
            for _, edges in uME_pos.items():
                SIT_value += calcEnjoyment(edges, t_obs, rate, method)
            ATGS_values['SIT'].append(SIT_value)

    minValues, maxValues = getATGSBounds(ATGS_values, minValues, maxValues)
    minValues["GIP"], maxValues["GIP"] = 0, 1 # Property 3
    minValues, maxValues = getGIDBounds(trimmed_PI_pos, t_obs, minValues, maxValues)

    return minValues, maxValues


if __name__ == "__main__":
    dataset_dir = "./Datasets/OSNs/"
    node_mapping_file = "./Datasets/Node_Mapping/C144_node_mapping.txt"
    dataset = "C144_attributed.txt"

    # Input parameters
    decay_method = "exp"
    decay_rate = 0.0001
    t_obs = 1672531150
    threshold_T = pow(10, -10)

    # Preprocess the dataset
    pro_dataset = preprocess_dataset(dataset_dir + dataset, t_obs)
    node_mapping = read_node_mapping(node_mapping_file)

    # Based on the dataset, construct an indexed structure
    start_t = findStartTime(t_obs, decay_rate, threshold_T)

    # Construct an indexed structure
    index, last_mutual_key, last_key = time_call("building the PANE-Index", buildPANEIndex, pro_dataset, node_mapping)
    print(f"Size of the PANE-Index: {asizeof.asizeof(index) / 1024**2:.2f} MB")

    # Trim the index to save space
    trimmed_index, mutual_nodes = time_call("trimming the PANE-Index", trimPANEIndex, index, start_t)
    print(f"Size of the trimmed PANE-Index: {asizeof.asizeof(trimmed_index) / 1024**2:.2f} MB")

    # Calculate the min-max values for each measure
    LB_values, UB_values = time_call("calculating the boundaries", findBoundsPANE, trimmed_index, mutual_nodes, t_obs, last_mutual_key, last_key, decay_method, decay_rate)
    print("Minimum values:", LB_values)
    print("Maximum values:", UB_values)

    
    # Store the min-max values in a txt file
    # output_file = dataset_dir + dataset.split('_attributed.txt')[0] + f"_bounds.txt"
    # with open(output_file, 'w') as f:
    #     f.write("Measure\tMin\tMax\n")
    #     for measure in LB_values.keys():
    #         f.write(f"{measure}\t{LB_values[measure]}\t{UB_values[measure]}\n")
