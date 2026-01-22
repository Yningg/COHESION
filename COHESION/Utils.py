
import numpy as np
import networkx as nx
import time
from collections import defaultdict


""" 
Utility functions for cohesiveness computation
"""

def time_decay(t_cur, t_i, rate, method):
    time_diff = t_cur - t_i

    if method == "exp":
        return np.exp(-rate * time_diff) 
    elif method == "poly":
        return 1 / ((time_diff + 1) ** rate)
        

# Calculate the excitation degree of node i up to time t in subgraph H
def excitation_degree(t, sentiment, activities, rate, method):
    degree = 1

    sentimental_activities = [activity for activity in activities if activity[1] != 0]
    if not sentimental_activities:
        return degree
        
    timestamp_list = np.array([activity[0] for activity in sentimental_activities])
    sentiment_list = np.array([activity[1] for activity in sentimental_activities])

    sign_values = np.sign(sentiment_list * sentiment) # Identify the polarity of the sentiment
    decay_values = np.array([time_decay(t, timestamp, rate, method) for timestamp in timestamp_list])

    degree += np.dot(sign_values, decay_values)
    
    return max(0, degree)


# Calculate the elicited sentiment Esenti
def ESenti(t, sentiment, previous_activities, rate, method):
    if sentiment == 0:
        return 0
    return sentiment * excitation_degree(t, sentiment, previous_activities, rate, method)


# Calculate the cumulative enjoyment given an edge set
def calcEnjoyment(edges, cur_t, rate, method):
    edge_values = np.array([ESenti(timestamp, sentiment, edges[:i], rate, method) * time_decay(cur_t, timestamp, rate, method) for i, (timestamp, sentiment) in enumerate(edges)])
    enjoyment = np.sum(edge_values)
    return enjoyment



# Find the start time based on the decay function and rate, also return the threshold value at that time
def findStartTime(t_obs, rate, threshold=pow(10, -10)):
    start_t = int(t_obs + np.log(threshold) / rate)
    return start_t
            

# Get edges related to user u
def getEdges(PI, u, neighbors):
    edge_list = []
    pairs = {(u, v) if u < v else (v, u) for v in neighbors}

    if not pairs:
        return edge_list
        
    for pair in pairs:
        edges = PI[pair][1]
        edge_list.extend(edges)

    edge_list.sort(key=lambda x: x[0]) # Sort once each by timestamp
   
    return edge_list


def getPairEdges(PI, u, neighbors):
    edge_dict = defaultdict(lambda: list)
    pairs = {(u, v) if u < v else (v, u) for v in neighbors}

    if not pairs:
        return edge_dict
        
    for pair in pairs:
        edges = PI[pair][1]
        if edges:
            edge_dict[pair] = edges
    
    # Sort edge lists according to the last timestamp
    edge_dict = dict(sorted(edge_dict.items(), key=lambda x: x[1][-1][0], reverse=True))
  
    return edge_dict


""" 
Utility functions for data processing and other operations
"""
def graph_construction(attribute_file, t_obs):
    starttime = time.time()
    attributed_G = nx.MultiDiGraph()
    
    with open(attribute_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            u, v, timestamp, sentiment = line.split()
            timestamp = int(timestamp)
            
            if timestamp <= t_obs:
                attributed_G.add_edge(u, v, timestamp=timestamp, sentiment=sentiment)

    endtime = time.time()
    print(f"Time taken for building the original graph: {endtime - starttime} seconds")
    print(f"Original graph info: {attributed_G.number_of_nodes()} nodes, {attributed_G.number_of_edges()} edges")

    return attributed_G


def preprocess_dataset(dataset_path, t_obs):
    preprocessed_edges = []
    with open(dataset_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            u_raw, v_raw, timestamp, sentiment = parts[0], parts[1], int(parts[2]), int(parts[3])
           
            if timestamp <= t_obs:
                preprocessed_edges.append((u_raw, v_raw, timestamp, sentiment))
    
    return preprocessed_edges


def time_call(label, func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    print(f"Time taken for {label}: {time.time() - start:.4f} seconds")
    return result


def normalize_scores(score, measure, LB_values, UB_values):
    min_val = LB_values[measure]
    max_val = UB_values[measure]
    if max_val == min_val: 
        return 0.0 
    return (score - min_val) / (max_val - min_val)


def read_node_mapping(node_mapping_file):
    node_mapping = {}
    with open(node_mapping_file, 'r') as f:
        lines = f.readlines()
        node_mapping = {int(line.split("\t")[0]): int(line.split("\t")[1]) for line in lines}
    
    return node_mapping