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
import networkx as nx


target_path = "./"
sys.path.append(target_path)
from COHESION.Utils import ESenti, time_decay, calcEnjoyment, graph_construction


# Get edges, and mutual edges related to user u
def getPosEdges(graph, u, cur_t):
    uE = list(graph.out_edges(u, data=True)) + list(graph.in_edges(u, data=True))
    uE_pos = [(edge[0], edge[1], int(edge[2]['timestamp']), 1) for edge in uE if int(edge[2]['timestamp']) <= cur_t]
    uE_pos = list(set(uE_pos))  # Remove duplicate edges
    uE_pos = sorted(uE_pos, key=lambda x: x[2])  # Sort by timestamp
    
    uME_pos = defaultdict(list)
    user_pairs = set([(u, v) for u, v, _, _ in uE_pos])
    mutual_pairs = {(min(pair[0], pair[1]), max(pair[0], pair[1])) for pair in user_pairs if (pair[1], pair[0]) in user_pairs and pair[0] != pair[1]}

    if mutual_pairs:
        for (u, v, t, s) in uE_pos:
            key = (u, v) if u < v else (v, u)
            if key in mutual_pairs:
                uME_pos[key].append((t, s))

    uE_pos = [(t, s) for (_, _, t, s) in uE_pos]

    return uE_pos, uME_pos

# Get the bounds of ATG-S measures
def getATGSBounds(ATGS_values, minValues, maxValues):
    maxValues['EI'], maxValues['SIT'] = round(max(ATGS_values['EI']), 4), round(max(ATGS_values['SIT']), 4)
    maxValues['CED'] = maxValues['EI']
    minValues['EI'], minValues['SIT'], minValues['CED'] = -maxValues['EI'], -maxValues['SIT'], -maxValues['CED']
    return minValues, maxValues


# Get the bounds of GI-S measures
def getGIDBounds(graph, cur_t, minValues, maxValues):
    minValues["GID"] = 0

    # Calculate the interaction number for each user pair in the graph
    interaction_count = defaultdict(int)
    for u, v in graph.edges():
        if u != v:
            interaction_count[(min(u, v), max(u, v))] += 1
    
    # Find the maximum interaction number among all user pairs
    max_interaction = max(interaction_count.values()) if interaction_count else 0
    maxValues["GID"] = round(max_interaction / (2 * cur_t), 4) if cur_t > 0 else 1

    return minValues, maxValues


def findBounds(graph, cur_t, method, rate):
    
    starttime = time.time()
    psyM = ['EI', 'SIT', 'CED', 'GIP', 'GID']
    minValues = {m: 0 for m in psyM}
    maxValues = {m: 0 for m in psyM}
    ATGS_values = {'EI': [], 'SIT': []}

    # Take extreme positive configuration as an example
    for u in tqdm(graph.nodes(), desc="Calculating boundaries for each user"):
        uE_pos, uME_pos = getPosEdges(graph, u, cur_t)

        # Calculate the Enjoyment Index (EI) of node u in subgraph H at time t_cur
        ESenti_values = np.array([ESenti(timestamp, sentiment, uE_pos[:i], rate, method) for i, (timestamp, sentiment) in enumerate(uE_pos)])
        time_decay_values = np.array([time_decay(cur_t, timestamp, rate, method) for (timestamp, _) in uE_pos])
        ATGS_values['EI'].append(np.dot(ESenti_values, time_decay_values))

        SIT_value = 0
        for _, edges in uME_pos.items():
            SIT_value += calcEnjoyment(edges, cur_t, rate, method)
        ATGS_values['SIT'].append(SIT_value)


    minValues, maxValues = getATGSBounds(ATGS_values, minValues, maxValues)
    minValues["GIP"], maxValues["GIP"] = 0, 1 # Property 3
    minValues, maxValues = getGIDBounds(graph, cur_t, minValues, maxValues)

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

    # Construct the graph based on the sliced edge
    G = graph_construction(dataset_dir + dataset)

    # Calculate the min-max values for each measure
    LB_values, UB_values, time_spent = findBounds(G, cur_t, decay_method, decay_rate)

    print("Minimum values:", LB_values)
    print("Maximum values:", UB_values)
    
    # Store the min-max values in a txt file
    output_file = dataset_dir + dataset.split('_attributed.txt')[0] + f"_bounds.txt"
    with open(output_file, 'w') as f:
        f.write("Measure\tMin\tMax\n")
        for measure in LB_values.keys():
            f.write(f"{measure}\t{LB_values[measure]}\t{UB_values[measure]}\n")
