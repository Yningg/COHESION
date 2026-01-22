"""
This script is used to find the boundaries of each measure given a graph (index method without trimming)

Input: Graph, decay_method, decay_rate
Output: min-max of five measures
"""

from collections import defaultdict
from tqdm import tqdm
import sys
import tracemalloc


target_path = "./"
sys.path.append(target_path)
from COHESION.Utils import calcEnjoyment, preprocess_dataset, time_call


# Build the PANE-Index for the graph
def buildPANEIndex(dataset):
    id_map = {} # map original node ids to continuous ids (save space)
    next_id = 0

    PI = defaultdict(list) # for each pair, store edge list
    pair_masks = defaultdict(int)   # for each pair, store direction bitmask
    NI = defaultdict(lambda: [set(), set()]) # for each node, store directed neighbors and mutual neighbors
    last_key = ()
    last_mutual_t, last_mutual_key = -1, ()

    for u_raw, v_raw, timestamp, sentiment in dataset:
        # Map IDs to contiguous integers
        if u_raw not in id_map:
            id_map[u_raw] = next_id
            u = next_id
            next_id += 1
        else:
            u = id_map[u_raw]

        if v_raw not in id_map:
            id_map[v_raw] = next_id
            v = next_id
            next_id += 1
        else:
            v = id_map[v_raw]

            
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


# Get edges related to user u
def getEdges(PI, u, neighbors):
    edge_list = []
    pairs = {(u, v) if u < v else (v, u) for v in neighbors}

    if not pairs:
        return edge_list
    for pair in pairs: 
        edge_list.extend(PI[pair])

    edge_list.sort(key=lambda x: x[0]) # Sort once each by timestamp
   
    return edge_list


def getPairEdges(PI, u, neighbors):
    edge_list = defaultdict(list)
    pairs = {(u, v) if u < v else (v, u) for v in neighbors}

    if not pairs:
        return edge_list
        
    for pair in pairs:
        edge_list[pair].extend(PI[pair])

    return edge_list


# Get the bounds of ATG-S measures
def getATGSBounds(ATGS_values, minValues, maxValues):
    maxValues['EI'], maxValues['SIT'] = round(max(ATGS_values['EI']), 4), round(max(ATGS_values['SIT']), 4)
    maxValues['CED'] = maxValues['EI']
    minValues['EI'], minValues['SIT'], minValues['CED'] = -maxValues['EI'], -maxValues['SIT'], -maxValues['CED']
    return minValues, maxValues


# Get the bounds of GI-S measures
def getGIDBounds(PI, t_obs, minValues, maxValues):
    minValues["GID"] = 0

    # Find the maximum interaction number among all user pairs
    max_interaction = 0
    for key, edges in PI.items():
        if key[0] != key[1]:
            edge_count = len(edges)
            if edge_count > max_interaction:
                max_interaction = edge_count

    maxValues["GID"] = max_interaction / (2 * t_obs if t_obs > 0 else 1)

    return minValues, maxValues


def findBoundsPANE(index, t_obs, method, rate):
    psyM = ['EI', 'SIT', 'CED', 'GIP', 'GID']
    minValues = {m: 0 for m in psyM}
    maxValues = {m: 0 for m in psyM}
    ATGS_values = {'EI': [], 'SIT': []}
    node_set = list(index["NI"].keys())

    # Take extreme positive configuration as an example
    PI_pos = {key: [(t, 1) for (t, _) in index["PI"][key]] for key in index["PI"]}
    for u in tqdm(node_set, desc="Calculating boundaries for each user"):
        uE_pos = getEdges(PI_pos, u, index["NI"][u][0] | index["NI"][u][1])
        uME_pos = getPairEdges(PI_pos, u, index["NI"][u][1])

        # Calculate the Enjoyment Index (EI) of node u in subgraph H at time t_obs
        ATGS_values['EI'].append(calcEnjoyment(uE_pos, t_obs, rate, method))

        SIT_value = 0
        for _, edges in uME_pos.items():
            SIT_value += calcEnjoyment(edges, t_obs, rate, method)
        ATGS_values['SIT'].append(SIT_value)
     

    minValues, maxValues = getATGSBounds(ATGS_values, minValues, maxValues)
    minValues["GIP"], maxValues["GIP"] = 0, 1 # Property 3
    minValues, maxValues = getGIDBounds(index["PI"], t_obs, minValues, maxValues)

    return minValues, maxValues



if __name__ == "__main__":
    dataset_dir = "./Datasets/OSNs/"
    dataset = "C144_attributed.txt"

    # Input parameters
    decay_method = "exp"
    decay_rate = 0.0001
    t_obs = 1672531150

    # Preprocess the dataset
    pro_dataset = preprocess_dataset(dataset_dir + dataset, t_obs)

    # Construct an indexed structure
    index, last_mutual_key, last_key = time_call("building the PANE-Index", buildPANEIndex, pro_dataset)

    # Calculate the min-max values for each measure
    LB_values, UB_values = time_call("calculating the boundaries", findBoundsPANE, index, t_obs, decay_method, decay_rate)
    print("Minimum values:", LB_values)
    print("Maximum values:", UB_values)
    
    # Store the min-max values in a txt file
    # output_file = dataset_dir + dataset.split('_attributed.txt')[0] + f"_bounds.txt"
    # with open(output_file, 'w') as f:
    #     f.write("Measure\tMin\tMax\n")
    #     for measure in LB_values.keys():
    #         f.write(f"{measure}\t{LB_values[measure]}\t{UB_values[measure]}\n")
