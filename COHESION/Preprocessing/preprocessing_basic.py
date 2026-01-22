"""
This script is used to find the boundaries of each measure given a graph (basic method)

Input: Graph, decay_method, decay_rate
Output: min-max of five measures
"""

from collections import defaultdict
from tqdm import tqdm
import sys
import networkx as nx


target_path = "./"
sys.path.append(target_path)
from COHESION.Utils import calcEnjoyment, graph_construction, time_call



# Get edges, and mutual edges related to user u
def getEdges(graph, u):
    uE = sorted({(s, t, d['timestamp'], int(d['sentiment'])) 
            for s, t, d in list(graph.out_edges(u, data=True)) + list(graph.in_edges(u, data=True))}, 
            key=lambda x: x[2])
    
    uME = defaultdict(list)
    user_pairs = set([(u, v) for u, v, _, _ in uE])
    mutual_pairs = {(min(pair[0], pair[1]), max(pair[0], pair[1])) for pair in user_pairs if (pair[1], pair[0]) in user_pairs and pair[0] != pair[1]}

    if mutual_pairs:
        for (u, v, t, s) in uE:
            key = (u, v) if u < v else (v, u)
            if key in mutual_pairs:
                uME[key].append((t, s))

    uE_pos = [(t, s) for (_, _, t, s) in uE]

    return uE_pos, uME

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


def findBounds(graph, t_obs, method, rate):

    psyM = ['EI', 'SIT', 'CED', 'GIP', 'GID']
    minValues = {m: 0 for m in psyM}
    maxValues = {m: 0 for m in psyM}
    ATGS_values = {'EI': [], 'SIT': []}

    # Take extreme positive configuration as an example
    pos_G = graph.copy()
    nx.set_edge_attributes(pos_G, 1, 'sentiment')
    for u in tqdm(graph.nodes(), desc="Calculating boundaries for each user"):
        uE, uME = getEdges(pos_G, u)

        ATGS_values['EI'].append(calcEnjoyment(uE, t_obs, rate, method))

        SIT_value = 0
        for _, edges in uME.items():
            SIT_value += calcEnjoyment(edges, t_obs, rate, method)
        ATGS_values['SIT'].append(SIT_value)


    minValues, maxValues = getATGSBounds(ATGS_values, minValues, maxValues)
    minValues["GIP"], maxValues["GIP"] = 0, 1
    minValues, maxValues = getGIDBounds(pos_G, t_obs, minValues, maxValues)

    return minValues, maxValues



if __name__ == "__main__":
    dataset_dir = "./Datasets/OSNs/"
    dataset = "CC_attributed.txt"

    # Input parameters
    decay_method = "exp"
    decay_rate = 0.0001
    t_obs = 1643673425

    # Construct the graph based on the sliced edge
    G = time_call("building the original graph", graph_construction, dataset_dir + dataset, t_obs)

    # Calculate the min-max values for each measure
    LB_values, UB_values = time_call("calculating boundaries", findBounds, G, t_obs, decay_method, decay_rate)
    print("Minimum values:", LB_values)
    print("Maximum values:", UB_values)
    
    # Store the min-max values in a txt file
    # output_file = dataset_dir + dataset.split('_attributed.txt')[0] + f"_bounds.txt"
    # with open(output_file, 'w') as f:
    #     f.write("Measure\tMin\tMax\n")
    #     for measure in LB_values.keys():
    #         f.write(f"{measure}\t{LB_values[measure]}\t{UB_values[measure]}\n")
