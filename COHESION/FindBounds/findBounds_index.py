"""
This script is used to find the boundaries of each measure given a graph (basic method)

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
from COHESION.Utils import calcEnjoyment


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


# Get edges, and mutual edges related to user u
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


# Get the bounds of ATG-S measures
def getATGSBounds(ATGS_values, minValues, maxValues):
    maxValues['EI'], maxValues['SIT'] = max(ATGS_values['EI']), max(ATGS_values['SIT'])
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

    maxValues["GID"] = max_interaction / (2 * cur_t if cur_t > 0 else 1)

    return minValues, maxValues


def findBoundsIndex(index, cur_t, method, rate):
    starttime = time.time()
    psyM = ['EI', 'SIT', 'CED', 'GIP', 'GID']
    minValues = {m: 0 for m in psyM}
    maxValues = {m: 0 for m in psyM}
    ATGS_values = {'EI': [], 'SIT': []}
    node_set = list(index["NI"].keys())

    # Take extreme positive configuration as an example
    for u in tqdm(node_set, desc="Calculating boundaries for each user"):
        uE_pos = getPosEdges(index, u, index["NI"][u]["dN"] | index["NI"][u]["mN"])
        uME_pos = getPosPairEdges(index, u, index["NI"][u]["mN"])

        # Calculate the Enjoyment Index (EI) of node u in subgraph H at time t_cur
        ATGS_values['EI'].append(calcEnjoyment(uE_pos, cur_t, rate, method))

        SIT_value = 0
        for _, info in uME_pos.items():
            SIT_value += calcEnjoyment(info["edges"], cur_t, rate, method)
        ATGS_values['SIT'].append(SIT_value)
     

    minValues, maxValues = getATGSBounds(ATGS_values, minValues, maxValues)
    minValues["GIP"], maxValues["GIP"] = 0, 1 # Property 3
    minValues, maxValues = getGIDBounds(index["PI"], cur_t, minValues, maxValues)

    endtime = time.time()
    time_spent = endtime - starttime
    print(f"Time taken for calculating boundaries: {time_spent} seconds")

    return minValues, maxValues, time_spent



if __name__ == "__main__":
    dataset_dir = "./Datasets/Networks/"
    dataset = "CC_attributed.txt"

    # Input parameters
    decay_method = "exp"
    decay_rate = 0.0001
    cur_t = 1643673425

    # Based on the dataset, construct an indexed structure to save the info
    index, last_mutual_t, last_t = buildPNIndex(dataset_dir + dataset, cur_t)

    # Calculate the min-max values for each measure
    LB_values, UB_values, time_spent = findBoundsIndex(index, cur_t, decay_method, decay_rate)

    print("Minimum values:", LB_values)
    print("Maximum values:", UB_values)
    
    # Store the min-max values in a txt file
    output_file = dataset_dir + dataset.split('_attributed.txt')[0] + f"_bounds.txt"
    with open(output_file, 'w') as f:
        f.write("Measure\tMin\tMax\n")
        for measure in LB_values.keys():
            f.write(f"{measure}\t{LB_values[measure]}\t{UB_values[measure]}\n")
