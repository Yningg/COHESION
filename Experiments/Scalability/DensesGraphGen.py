import random
import networkx as nx
import tqdm
import numpy as np


def graph_stats(attributed_G): 
    num_nodes = attributed_G.number_of_nodes()
    num_edges = attributed_G.number_of_edges()
    num_timestamps = len(set([int(d['timestamp']) for u, v, d in attributed_G.edges(data=True)]))
    density = nx.density(attributed_G)
    average_degree = sum(dict(attributed_G.degree()).values()) / num_nodes
    print(f"{num_nodes:,} & {num_edges:,} & {num_timestamps:,} & {density:.4f} & {average_degree:.2f}\\\\")


def extract_time_deltas(attribute_file):
    timestamps = []
    with open(attribute_file, 'r') as f:
        for line in f:
            u, v, t = line.strip().split(" ")
            timestamps.append(int(t))
    
    timestamps = sorted(timestamps)
    deltas = np.diff(timestamps)

    return deltas


def sample_empirical_delta(deltas):
    return random.choice(deltas)


# Generate a directed multigraph with density > given threshold
def generate_dense_multidigraph(n_nodes, density, time_deltas):
    t_cur = 0
    graph_edges = []

    # Maximum edges for directed graph without self-loops
    max_edges = n_nodes * (n_nodes - 1)

    # Target edge count
    target_edges = int(density * max_edges) + 1

    # Add edges
    for i in tqdm.tqdm(range(target_edges)):
        u = random.randrange(n_nodes)
        v = random.randrange(n_nodes)
        
        delta_t = sample_empirical_delta(time_deltas)
        t_cur += delta_t
        sentiment = random.choice([-1, 0, 1])
        graph_edges.append((u, v, t_cur, sentiment))

    print(f"Generated {len(graph_edges)} edges for a graph with {n_nodes} nodes and target density {density}.")
    return graph_edges


def outputGraph(edges, output_file):
    with open(output_file, 'w') as f:
        for u, v, t, sentiment in edges:
            f.write(f"{u}\t{v}\t{t}\t{sentiment}\n")


if __name__ == "__main__":
    # Generate the empirical distribution of timestamp differences from the SuperUser dataset
    target_dir = "./Datasets/SyntheticOSNs/"
    attribute_file ="D:/TemporalNetworks/SuperUser.txt"

    time_deltas = extract_time_deltas(attribute_file)

    for density in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        edges = generate_dense_multidigraph(1000, density, time_deltas)
        outputGraph(edges, target_dir + f"Dense_Multidigraph_{density}.txt")