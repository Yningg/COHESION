
import numpy as np
import networkx as nx
import time
from functools import wraps
from collections import defaultdict
import tqdm
import bisect


# Define global variables for the start time of each dataset
DATASET_START_TIMES = {
    "BTW": 1332960223,
    "CC": 1321675332,
    "C26": 1669853182,
    "C144": 1669853287,
    "AskUbuntu": 1231431607,
    "MathOverflow": 1254192988,
    "StackOverflow": 1217567877,
    "SuperUser": 1217651565
}


""" 
Utility functions for general usage
"""
def time_call(label=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                name = label or func.__name__
                print(f"[{name}] {duration:.6f} seconds")
        return wrapper
    return decorator


""" 
Utility functions for cohesiveness computation
"""
# Build the PANE-Index for the graph
@time_call("building the PANE-Index")
def buildPANEIndex(dataset):
    PI = defaultdict(list) # for each pair, store edge list
    pair_masks = defaultdict(int)   # for each pair, store direction bitmask
    NI = defaultdict(lambda: [set(), set()]) # for each node, store directed neighbors and mutual neighbors
    last_key = ()
    last_mutual_t, last_mutual_key = -1, ()

    for u, v, timestamp, sentiment in tqdm.tqdm(dataset):
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

    
    for (u, v), edges in tqdm.tqdm(PI.items()):
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



@time_call("trimming the PANE-Index")
def trimPANEIndex(index, start_t):
    trimmed_PI = {}
    mutual_nodes = set()
    removed_pairs = set()

    for (u, v), edges in index["PI"].items():
        edge_count = len(edges)
        valid_edges = []
        if edges[-1][0] < start_t:
            removed_pairs.add((u, v))
            valid_edges = []
        elif edges[0][0] >= start_t:
            valid_edges = edges
        else:
            idx = bisect.bisect_left([edge[0] for edge in edges], start_t)
            valid_edges = edges[idx:]
        
        trimmed_PI[(u, v)] = [edge_count, valid_edges]

        if index["NI"][u][1]:
            mutual_nodes.add(u)
        if index["NI"][v][1]:
            mutual_nodes.add(v)

    return {"PI": trimmed_PI, "NI": index["NI"]}, mutual_nodes



def time_decay(t_obs, t_i, method, rate):
    time_diff = t_obs - t_i

    if method == "exp":
        return np.exp(-rate * time_diff) 
    elif method == "poly":
        return 1 / ((time_diff + 1) ** rate)
        

# Calculate the excitation degree of node i up to time t in subgraph H
def excitation_degree(t, sentiment, activities, method, rate):
    degree = 1

    if not activities:
        return degree
    
    arr = np.array(activities)
    mask = arr[:, 1] != 0
    if not mask.any():
        return degree
        
    timestamp_list = arr[mask, 0]
    sentiment_list = arr[mask, 1]

    sign_values = np.sign(sentiment_list * sentiment) # Identify the polarity of the sentiment
    decay_values = time_decay(t, timestamp_list, method, rate)

    degree += np.dot(sign_values, decay_values)
    
    return max(0, degree)


# Calculate the elicited sentiment Esenti
def ESenti(t, sentiment, previous_activities, method, rate):
    if sentiment == 0:
        return 0
    return sentiment * excitation_degree(t, sentiment, previous_activities, method, rate)


# Calculate the cumulative enjoyment given an edge set
def calcEnjoyment(edges, t_obs, method, rate):
    if not edges:
        return 0
    edge_values = np.array([ESenti(timestamp, sentiment, edges[:i], method, rate) * time_decay(t_obs, timestamp, method, rate) for i, (timestamp, sentiment) in enumerate(edges)])
    enjoyment = np.sum(edge_values)
    return enjoyment


# Find the start time based on the decay function and rate, also return the threshold value at that time
def findStartTime(t_obs, rate, threshold=pow(10, -10)):
    start_t = int(t_obs + np.log(threshold) / rate)
    return start_t
            

# Get edges related to user u
def getEdges(PI, u, neighbors, trim):
    edge_list = []
    edge_count = 0
    pairs = {(u, v) if u < v else (v, u) for v in neighbors}

    if not pairs:
        return edge_list, edge_count
    for pair in pairs: 
        if pair in PI:
            if trim:
                edge_list.extend(PI[pair][1])
                edge_count += PI[pair][0]
            else:
                edge_list.extend(PI[pair])
                edge_count += len(PI[pair])

    edge_list.sort(key=lambda x: x[0]) # Sort once each by timestamp
   
    return edge_list, edge_count


def getPairEdges(PI, u, neighbors, trim, sort):
    pairs = {(u, v) if u < v else (v, u) for v in neighbors}

    if not pairs:
        return {}
    
    idx = 1 if trim else slice(None)
    edge_list = {
        pair: PI[pair][idx]
        for pair in pairs
        if pair in PI and PI[pair][idx]
    }

    if sort:
        edge_list = dict(sorted(edge_list.items(), key=lambda x: x[1][-1][0], reverse=True))

    return edge_list


# Get the bounds of ATG-S measures
def getATGSBounds(minValues, maxValues):
    maxValues['EI'], maxValues['SIT'] = round(maxValues['EI'], 4), round(maxValues['SIT'], 4)
    minValues['EI'], minValues['SIT'] = -maxValues['EI'], -maxValues['SIT']
    minValues['CED'], maxValues['CED'] = minValues['EI'], maxValues['EI']
    return minValues, maxValues


# Get the bounds of GI-S measures
def getGIDBounds(PI, t_obs, minValues, maxValues, trim=False):
    minValues["GID"] = 0

    # Find the maximum interaction number among all user pairs
    max_interaction = 0
    if trim:
        for key, (count, _) in PI.items():
            if key[0] != key[1]:
                edge_count = count
                if edge_count > max_interaction:
                    max_interaction = edge_count
    else:
        for key, edges in PI.items():
            if key[0] != key[1]:
                edge_count = len(edges)
                if edge_count > max_interaction:
                    max_interaction = edge_count

    maxValues["GID"] = round(max_interaction / (2 * t_obs), 4) if t_obs > 0 else 0

    return minValues, maxValues


def normalize_scores(score, measure, LB_values, UB_values):
    min_val = LB_values[measure]
    max_val = UB_values[measure]
    if max_val == min_val: 
        return 0.0 
    return (score - min_val) / (max_val - min_val)


""" 
Utility functions for data processing and other operations
"""
@time_call("building the original graph")
def graph_construction(attribute_file, dataset, t_obs):
    attributed_G = nx.MultiDiGraph()
    delta = t_obs - DATASET_START_TIMES[dataset]
    
    with open(attribute_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            u, v, timestamp, sentiment = line.split()
            timestamp = int(timestamp) - DATASET_START_TIMES[dataset]
            
            if timestamp <= delta:
                attributed_G.add_edge(u, v, timestamp=timestamp, sentiment=sentiment)

    print(f"Original graph info: {attributed_G.number_of_nodes()} nodes, {attributed_G.number_of_edges()} edges")

    return attributed_G, delta


# Get all edges by t_obs
def preprocess_dataset(dataset_path, dataset, node_mapping, t_obs):
    preprocessed_edges = []
    start_t = DATASET_START_TIMES[dataset]
    delta = t_obs - start_t
    with open(dataset_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            u_raw, v_raw, timestamp_delta, sentiment = parts[0], parts[1], int(parts[2]) - start_t, int(parts[3])
            if timestamp_delta <= delta:
                preprocessed_edges.append((node_mapping[int(u_raw)], node_mapping[int(v_raw)], timestamp_delta, sentiment))
    
    return preprocessed_edges, delta


def read_node_mapping(node_mapping_file):
    node_mapping = {}
    with open(node_mapping_file, 'r') as f:
        lines = f.readlines()
        node_mapping = {int(line.split("\t")[0]): int(line.split("\t")[1]) for line in lines}
    
    return node_mapping
