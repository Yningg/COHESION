"""
This script is used to find the boundaries of each measure given a graph (index method without trimming)

Input: Graph, decay_method, decay_rate
Output: min-max of five measures
"""

from tqdm import tqdm
import sys
import numpy as np

target_path = "./"
sys.path.append(target_path)
from COHESION.Utils import read_node_mapping, preprocess_dataset, buildPANEIndex, calcEnjoyment, getATGSBounds, getGIDBounds, getEdges, getPairEdges, time_call


@time_call("finding the bounds of each measure")
def findBoundsPANE(index, t_obs, method, rate):
    psyM = ['EI', 'SIT', 'CED', 'GIP', 'GID']
    minValues = {m: 0 for m in psyM}
    maxValues = {m: 0 for m in psyM}
    node_set = list(index["NI"].keys())

    # Take extreme positive configuration as an example
    PI, NI = index["PI"], index["NI"]
    PI_pos = {key: [(t, 1) for (t, _) in PI[key]] for key in PI}
    for u in tqdm(node_set, desc="Calculating boundaries for each user"):
        uE_pos, _ = getEdges(PI_pos, u, list(NI[u][0]) + list(NI[u][1]), trim=False)
        uME_pos = getPairEdges(PI_pos, u, list(NI[u][1]), trim=False, sort=False)

        # Calculate the Enjoyment Index (EI) of node u in subgraph H at time t_obs
        EI_value = calcEnjoyment(uE_pos, t_obs, method, rate)
        maxValues['EI'] = max(maxValues['EI'], EI_value)

        SIT_value = 0
        for _, edges in uME_pos.items():
            val = calcEnjoyment(edges, t_obs, method, rate)
            SIT_value += val if not np.isnan(val) else 0
        maxValues['SIT'] = max(maxValues['SIT'], SIT_value)

    minValues, maxValues = getATGSBounds(minValues, maxValues)
    minValues["GIP"], maxValues["GIP"] = 0, 1 # Property 3
    minValues, maxValues = getGIDBounds(PI, t_obs, minValues, maxValues, trim=False)

    return minValues, maxValues


if __name__ == "__main__":
    # Input parameters
    decay_method = "exp"
    decay_rate = 0.0001
    t_obs = 1672531150

    # Preprocess the dataset
    node_mapping_file = "./Datasets/Node_Mapping/C144_node_mapping.txt"
    node_mapping = read_node_mapping(node_mapping_file)
    pro_dataset, t_obs = preprocess_dataset("./Datasets/OSNs/C144_attributed.txt", "C144", node_mapping, t_obs)

    # Construct an indexed structure
    index, last_mutual_key, last_key = buildPANEIndex(pro_dataset)

    # Calculate the min-max values for each measure
    LB_values, UB_values = findBoundsPANE(index, t_obs, decay_method, decay_rate)
    print("Minimum values:", LB_values)
    print("Maximum values:", UB_values)
    
    # Store the min-max values in a txt file
    # output_file = dataset_dir + dataset.split('_attributed.txt')[0] + f"_bounds.txt"
    # with open(output_file, 'w') as f:
    #     f.write("Measure\tMin\tMax\n")
    #     for measure in LB_values.keys():
    #         f.write(f"{measure}\t{LB_values[measure]}\t{UB_values[measure]}\n")
