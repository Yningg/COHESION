"""
This script is used to find the boundaries of each measure given a graph (basic method)

Input: Graph, decay_method, decay_rate
Output: min-max of five measures
"""

from collections import defaultdict, Counter
from itertools import chain
from tqdm import tqdm
import sys
import numpy as np
import networkx as nx


target_path = "./"
sys.path.append(target_path)
from COHESION.Utils import calcEnjoyment, graph_construction, time_call, getATGSBounds



# Get edges, and mutual edges related to user u
def getEdges(graph, u):
    all_edges = chain(graph.out_edges(u, data=True), graph.in_edges(u, data=True))
    uE = sorted({(s, t, d['timestamp'], int(d['sentiment'])) for s, t, d in all_edges}, 
            key=lambda x: x[2])
    
    directed_pairs = {(s, t) for s, t, _, _ in uE}
    mutual_pairs = {(min(s, t), max(s, t)) for s, t in directed_pairs if s != t and (t, s) in directed_pairs}
    
    uME = defaultdict(list)
    for (u, v, t, s) in uE:
        key = (u, v) if u < v else (v, u)
        if key in mutual_pairs:
            uME[key].append((t, s))

    uE_pos = [(t, s) for (_, _, t, s) in uE]

    return uE_pos, uME


# Get the bounds of GI-S measures
def getGIDBounds(graph, t_obs, minValues, maxValues):
    minValues["GID"] = 0

    # Calculate the interaction number for each user pair in the graph
    interaction_count = Counter((min(u, v), max(u, v)) for u, v in graph.edges() if u != v)
    
    # Find the maximum interaction number among all user pairs
    max_interaction = max(interaction_count.values(), default=0)
    maxValues["GID"] = round(max_interaction / (2 * t_obs), 4)

    return minValues, maxValues


@time_call("finding the bounds of each measure")
def findBounds(graph, t_obs, method, rate):

    psyM = ['EI', 'SIT', 'CED', 'GIP', 'GID']
    minValues = {m: 0 for m in psyM}
    maxValues = {m: 0 for m in psyM}

    # Take extreme positive configuration as an example
    nx.set_edge_attributes(graph, 1, 'sentiment')
    for u in tqdm(graph.nodes(), desc="Calculating boundaries for each user"):
        uE, uME = getEdges(graph, u)

        EI_value = calcEnjoyment(uE, t_obs, method, rate)
        maxValues['EI'] = max(maxValues['EI'], EI_value)

        SIT_value = 0
        for _, edges in uME.items():
            val = calcEnjoyment(edges, t_obs, method, rate)
            SIT_value += val if not np.isnan(val) else 0
        maxValues['SIT'] = max(maxValues['SIT'], SIT_value)

    maxValues['EI'], maxValues['SIT'] = round(maxValues['EI'], 4), round(maxValues['SIT'], 4)
    minValues['EI'], minValues['SIT'] = -maxValues['EI'], -maxValues['SIT']
    minValues['CED'], maxValues['CED'] = minValues['EI'], maxValues['EI']

    minValues, maxValues = getATGSBounds(minValues, maxValues)
    minValues["GIP"], maxValues["GIP"] = 0, 1
    minValues, maxValues = getGIDBounds(graph, t_obs, minValues, maxValues)

    return minValues, maxValues



if __name__ == "__main__":
    dataset_dir = "./Datasets/OSNs/"
    dataset = "CC_attributed.txt"

    # Input parameters
    decay_method = "exp"
    decay_rate = 0.0001
    t_obs = 1643673425

    # Construct the graph based on the sliced edge
    G, t_obs = graph_construction(dataset_dir + dataset, dataset, t_obs)

    # Calculate the min-max values for each measure
    LB_values, UB_values = findBounds(G, t_obs, decay_method, decay_rate)
    print("Minimum values:", LB_values)
    print("Maximum values:", UB_values)
    
    # Store the min-max values in a txt file
    # output_file = dataset_dir + dataset.split('_attributed.txt')[0] + f"_bounds.txt"
    # with open(output_file, 'w') as f:
    #     f.write("Measure\tMin\tMax\n")
    #     for measure in LB_values.keys():
    #         f.write(f"{measure}\t{LB_values[measure]}\t{UB_values[measure]}\n")
