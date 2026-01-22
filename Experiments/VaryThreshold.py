import sys
import networkx as nx
import tqdm

target_path = "./"
sys.path.append(target_path)
import COHESION.Preprocessing.preprocessing_index_trim as pp_it


# Input parameters
decay_method = "exp"
decay_rate = 0.0001
dataset_list = ["BTW", "CC", "C26", "C144"]
threshold_list = [pow(10, -50), pow(10, -40), pow(10, -30), pow(10, -20), pow(10, -10)]
data_path = "./Datasets/Networks/"
store_path = "./Datasets/Time_Spent/"
last_timestamps = {"BTW": 1506315747, "CC": 1643673425, "C26": 1672531185, "C144": 1672531150}


for dataset in dataset_list:
    dataset_path = data_path + dataset + "_attributed.txt"
    time_spent_list = []

    index, last_mutual_t, last_t = pp_it.buildPANEIndex(dataset_path, last_timestamps[dataset])
    
    for th in threshold_list:
        print(f"Processing dataset {dataset} with threshold {th}")
        start_t = pp_it.findStartTime(last_timestamps[dataset], decay_rate, th)
        LB_values, UB_values, trimmed_index, time_spent = pp_it.findBoundsPANE(index, start_t, last_timestamps[dataset], last_mutual_t, last_t, decay_method, decay_rate)

        print("Minimum values:", LB_values)
        print("Maximum values:", UB_values)
        time_spent_list.append(time_spent)

        # Store the min-max values in a txt file
        output_file = store_path + dataset.split('_attributed.txt')[0] + f"_indexpbounds_{threshold_list.index(th)}.txt"
        
        with open(output_file, 'w') as f:
            f.write("Measure\tMin\tMax\n")
            for measure in LB_values.keys():
                f.write(f"{measure}\t{LB_values[measure]}\t{UB_values[measure]}\n")
            f.write(f"Time_Spent\t{time_spent}\n")
            f.write(f"Threshold\t{th}\n")
    
    print(f"Time spent for dataset {dataset} with different thresholds: {time_spent_list}")