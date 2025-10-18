import numpy as np
import networkx as nx
import time

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


def graph_construction(attribute_file):
    starttime = time.time()
    attributed_G = nx.read_edgelist(attribute_file, nodetype=str, data=(('timestamp', str), ('sentiment', str)), create_using=nx.MultiDiGraph()) # type: ignore
    endtime = time.time()
    print(f"Time taken for building the original graph: {endtime - starttime} seconds")
    print(f"Original graph info: {attributed_G.number_of_nodes()} nodes, {attributed_G.number_of_edges()} edges")

    return attributed_G