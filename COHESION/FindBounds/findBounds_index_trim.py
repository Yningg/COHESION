"""
This script is used to find the boundaries of each measure given a graph (index method with trimming)

Input: Graph, decay_method, decay_rate
Output: min-max of five measures
"""

import time
from collections import defaultdict
from tqdm import tqdm
import sys
import numpy as np
import csv


target_path = "./"
sys.path.append(target_path)
from COHESION.Utils import ESenti, time_decay, calcEnjoyment


# Build an index for the graph
def buildPNIndex(file_path, cur_t):
    starttime = time.time()
    PI = defaultdict(lambda: {"edges": [], "lastT": -1})
    NI = defaultdict(lambda: {"dN": set(), "mN": set()})
    pairs = set()
    last_mutual_ts = []
    
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for u, v, timestamp, sentiment in reader:
            timestamp, sentiment = int(timestamp), int(sentiment)
            if timestamp > cur_t:
                break
            pair_key = (u, v) if u < v else (v, u)
            PI[pair_key]["edges"].append((timestamp, sentiment))
            pairs.add((u, v)) 
            
    # Classify relationships and build adjacency list
    for key, info in PI.items():
        u, v = key
        if u == v:
            NI[u]["dN"].add(v)
        else:
            if (u, v) in pairs and (v, u) in pairs:
                NI[u]["mN"].add(v)
                NI[v]["mN"].add(u)
                last_mutual_ts.append(PI[key]["edges"][-1][0])
            else:
                NI[u]["dN"].add(v)
                NI[v]["dN"].add(u)

        info["lastT"] = info["edges"][-1][0]
    
    # Sort PI by the "lastT" of each pair in descending order
    PI = dict(sorted(PI.items(), key=lambda item: item[1]["lastT"]))
    last_t = list(PI.values())[-1]['lastT'] if PI else -1
    last_mutual_t = max(last_mutual_ts) if last_mutual_ts else -1
    endtime = time.time()
    
    print(f"Time taken for building the index: {endtime - starttime} seconds")
    print(f"Last mutual interaction timestamp: {last_mutual_t}")
    print(f"Last overall interaction timestamp: {last_t}")
    return {"PI": PI, "NI": NI}, last_mutual_t, last_t



# Find the start time based on the decay function and rate, also return the threshold value at that time
def findStartTime(cur_t, rate, threshold=pow(10, -10)):
    start_t = int(cur_t + np.log(threshold) / rate)

    return start_t
            

# Get edges related to user u
def getPosEdges(index, u, neighbors):
    edge_pos = []
    pairs = {(u, v) if u < v else (v, u) for v in neighbors}

    if not pairs:
        return edge_pos
        
    for pair in pairs:
        edges = index["PI"][pair]["edges"]
        edges_pos = [(t, 1) for (t, _) in edges]
        edge_pos.extend(edges_pos)

    edge_pos.sort(key=lambda x: x[0]) # Sort once each by timestamp
   
    return edge_pos


def getPosPairEdges(index, u, neighbors):
    edge_pos = defaultdict(lambda: {"edges": [], "lastT": -1})
    pairs = {(u, v) if u < v else (v, u) for v in neighbors}

    if not pairs:
        return edge_pos
        
    for pair in pairs:
        edges = index["PI"][pair]["edges"]
        edges_pos = [(t, 1) for (t, _) in edges]
        edge_pos[pair]["edges"] = edges_pos
        edge_pos[pair]["lastT"] = index["PI"][pair]["lastT"]

    edge_pos = dict(sorted(edge_pos.items(), key=lambda item: item[1]["lastT"], reverse=True))
   
    return edge_pos


# Calculate the threshold for EI
def calcThresholdEI(index, node_set, last_t, cur_t, method, rate):
    # Find the pair with last timestamp
    key = [k for k, info in index["PI"].items() if info["lastT"] == last_t]
    u, v = key[0]
    uE_EI, vE_EI = -float('inf'), -float('inf')

    # get edge sets for both node in the pair
    if u == v:
        uE_pos = getPosEdges(index, u, index["NI"][u]["dN"] | index["NI"][u]["mN"])
        uE_EI = calcEnjoyment(uE_pos, cur_t, rate, method)
        node_set.remove(u)
        return node_set, uE_EI
 
    vE_pos = getPosEdges(index, v, index["NI"][v]["dN"] | index["NI"][v]["mN"])
    vE_EI = calcEnjoyment(vE_pos, cur_t, rate, method)
    node_set.remove(u)
    node_set.remove(v)

    return node_set, max(uE_EI, vE_EI)


# Calculate the threshold for SIT
def calcThresholdSIT(index, node_mutual_set, last_mutual_t, cur_t, method, rate):
    # Find the pair with last mutual timestamp
    key = [k for k, v in index["PI"].items() if v["lastT"] == last_mutual_t]
    u, v = key[0]
    uME_SIT, vME_SIT = -float('inf'), -float('inf')

    # get mutual edge sets for both node in the pair
    uME_pos = getPosPairEdges(index, u, index["NI"][u]["mN"])
    vME_pos = getPosPairEdges(index, v, index["NI"][v]["mN"])
    
    uME_SIT, vME_SIT = 0, 0
    for _, info in uME_pos.items():
        uME_SIT += calcEnjoyment(info["edges"], cur_t, rate, method)
    for _, info in vME_pos.items():
        vME_SIT += calcEnjoyment(info["edges"], cur_t, rate, method)
    
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
def getGIDBounds(PI, cur_t, minValues, maxValues):
    minValues["GID"] = 0

    # Find the maximum interaction number among all user pairs
    max_interaction = 0
    for key, info in PI.items():
        if key[0] != key[1]:
            edge_count = len(info["edges"])
            if edge_count > max_interaction:
                max_interaction = edge_count

    maxValues["GID"] = round(max_interaction / (2 * cur_t), 4) if cur_t > 0 else 1

    return minValues, maxValues


def trimIndex(index, start_t):
    starttime = time.time()

    trimmed_index = {"PI": defaultdict(lambda: {"edges": [], "lastT": -1}), "NI": defaultdict(lambda: {"dN": set(), "mN": set()})}
    removed_pairs = set()
    mutual_nodes = set() 

    for key, info in index["PI"].items():
        if info["lastT"] < start_t:
            removed_pairs.add(key)
            continue
        trimmed_index["PI"][key]["edges"] = [(t, s) for (t, s) in info["edges"] if t >= start_t]
        trimmed_index["PI"][key]["lastT"] = index["PI"][key]["lastT"]

    # Remove nodes in index["NI"] that have no edges in trimmed_index["PI"] 
    for u, neighbors in index["NI"].items():
        directed = {v for v in neighbors["dN"] if not ((u, v) in removed_pairs or (v, u) in removed_pairs)}
        mutual = {v for v in neighbors["mN"] if not ((u, v) in removed_pairs or (v, u) in removed_pairs)}
        if directed or mutual:
            trimmed_index["NI"][u]["dN"] = directed
            trimmed_index["NI"][u]["mN"] = mutual
            if mutual:
                mutual_nodes.add(u)
    
    endtime = time.time()
    print(f"Time taken for trimming the index: {endtime - starttime} seconds")

    return trimmed_index, list(trimmed_index["NI"].keys()), mutual_nodes


def findBoundsIndex(index, start_t, cur_t, last_mutual_t, last_t, method, rate):
    starttime = time.time()
    psyM = ['EI', 'SIT', 'CED', 'GIP', 'GID']
    minValues = {m: 0 for m in psyM}
    maxValues = {m: 0 for m in psyM}
    ATGS_values = {'EI': [], 'SIT': []}
    

    minValues["GIP"], maxValues["GIP"] = 0, 1 # Property 3
    minValues, maxValues = getGIDBounds(index["PI"], cur_t, minValues, maxValues)

    trimmed_index, nodes, mutual_nodes = trimIndex(index, start_t)

    nodes, EI_threshold = calcThresholdEI(trimmed_index, nodes, last_t, cur_t, method, rate)
    mutual_nodes, SIT_threshold = calcThresholdSIT(trimmed_index, mutual_nodes, last_mutual_t, cur_t, method, rate)
    ATGS_values['EI'].append(EI_threshold)
    ATGS_values['SIT'].append(SIT_threshold)

    # Take extreme positive configuration as an example
    # Separate the boundary calculation for EI and SIT to optimize performance
    for u in tqdm(nodes, desc="Calculating EI boundaries for each user"):
        uE_pos = getPosEdges(trimmed_index, u, trimmed_index["NI"][u]["dN"] | trimmed_index["NI"][u]["mN"])
        max_edge_EI = ESenti(uE_pos[-1][0], uE_pos[-1][1], uE_pos[:-1], rate, method) * time_decay(cur_t, uE_pos[-1][0], rate, method)
    
        if max_edge_EI * len(uE_pos) >= EI_threshold:  # A quick check to skip unnecessary EI calculations
            ATGS_values['EI'].append(calcEnjoyment(uE_pos, cur_t, rate, method))
        
    for u in tqdm(mutual_nodes, desc="Calculating SIT boundaries for each user"):
        uME_pos = getPosPairEdges(trimmed_index, u, trimmed_index["NI"][u]["mN"])
        max_edge_SIT = calcEnjoyment(list(uME_pos.values())[0]["edges"], cur_t, rate, method)

        if max_edge_SIT * len(uME_pos) >= SIT_threshold:  # A quick check to skip unnecessary SIT calculations       
            SIT_value = 0
            for _, info in uME_pos.items():
                SIT_value += calcEnjoyment(info["edges"], cur_t, rate, method)
            ATGS_values['SIT'].append(SIT_value)


    minValues, maxValues = getATGSBounds(ATGS_values, minValues, maxValues)

    endtime = time.time()
    time_spent = endtime - starttime
    print(f"Time taken for calculating boundaries: {time_spent} seconds")

    return minValues, maxValues, trimmed_index, time_spent


if __name__ == "__main__":
    dataset_dir = "./Datasets/Networks/"
    dataset = "C26_attributed.txt"

    # Input parameters
    decay_method = "exp"
    decay_rate = 0.0001
    cur_t = 1672531185
    threshold_T = pow(10, -10)
    start_t = findStartTime(cur_t, decay_rate, threshold_T)

    # Based on the dataset, construct an indexed structure to save the info
    index, last_mutual_t, last_t = buildPNIndex(dataset_dir + dataset, cur_t)

    # Calculate the min-max values for each measure
    LB_values, UB_values, trimmed_index, time_spent = findBoundsIndex(index, start_t, cur_t, last_mutual_t, last_t, decay_method, decay_rate)

    print("Minimum values:", LB_values)
    print("Maximum values:", UB_values)
    
    # Store the min-max values in a txt file
    output_file = dataset_dir + dataset.split('_attributed.txt')[0] + f"_bounds.txt"
    with open(output_file, 'w') as f:
        f.write("Measure\tMin\tMax\n")
        for measure in LB_values.keys():
            f.write(f"{measure}\t{LB_values[measure]}\t{UB_values[measure]}\n")
