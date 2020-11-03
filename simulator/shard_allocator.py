import numpy as np
import pandas as pd
import sys
import numpy as np
import math


def shard_allocator():

    if (not algorithm in ["random", "sequential", "SALP"]):
        sys.exit("Pass one of allocation algorithms: random/sequential/SALP as third param.")

    if (num_of_shards < 100 * num_of_nodes):
        sys.exit("There should be at least 100 times more shards than nodes.")

    if (algorithm == "random"):
        random_allocation()

    if (algorithm == "sequential"):
        sequential_allocation()

    if (algorithm == "SALP"):
        SALP_allocation()


def random_allocation():
    shards_on_nodes = []
    current_node = 1

    shards_shuffled = np.random.choice(range(1, num_of_shards + 1), num_of_shards, replace=False)

    for i in range(len(shards_shuffled)):
        shard = shards_shuffled[i]

        shards_on_nodes.append([shard, current_node])

        if ((i + 1) % 100 == 0):
            current_node += 1

    shards_on_nodes_df = pd.DataFrame(shards_on_nodes, columns=["shard", "node"])
    shards_on_nodes_df.sort_values('shard').to_csv("./simulator/shard_allocated.csv", index=False)


def sequential_allocation():
    shards_on_nodes = []
    current_node = 1

    for shard in range(num_of_shards):

        shards_on_nodes.append([shard + 1, current_node])

        if ((shard + 1) % 100 == 0):
            current_node += 1

    shards_on_nodes_df = pd.DataFrame(shards_on_nodes, columns=["shard", "node"])
    shards_on_nodes_df.to_csv("./simulator/shard_allocated.csv", index=False)


def SALP_allocation():
    load_vectors_df = pd.read_csv("./generator/load_vectors.csv", header=None)
    # print(load_vectors_df)
    WTS = load_vectors_df.sum(axis=0)
    # print("WTS: \n", WTS)
    periods_in_vector = load_vectors_df.shape[1]

    NWTS = WTS / num_of_nodes
    # print("NWTS: \n", NWTS)

    NWTS_module = calculate_vector_module(NWTS)
    # print(NWTS_module)

    modules_list = []

    current_shard = 0

    for index, row in load_vectors_df.iterrows():
        current_shard = current_shard + 1
        modules_list.append([calculate_vector_module(row), current_shard])


    modules_sorted_df = pd.DataFrame(modules_list, columns=["module", "shard"]).sort_values('module')
    list_load_vectors = load_vectors_df.values.tolist()

    list_inactive_nodes = []

    # shards_in_node = [[] for _ in range(num_of_nodes) ] # FSJ
    # loads_in_node = [[0] * periods_in_vector for _ in range(num_of_nodes)] # WSJ
    # print(loads_in_node)
    zeros_list = [0] * periods_in_vector
    nodes_detail = []
    for node in range(num_of_nodes):
        nodes_detail.append([node + 1, [], zeros_list])

    nodes_detail_df = pd.DataFrame(nodes_detail, columns=['node', 'shards', 'load_vector'])
    i = 0
    for shard in modules_sorted_df['shard']:
        # break
        node = calculate_node_for_shard(NWTS, nodes_detail_df['load_vector'].tolist(), list_load_vectors[shard - 1], list_inactive_nodes)

        shards_list = nodes_detail_df[nodes_detail_df.node == node]['shards'].item()
        shards_list.append(shard)

        node_load = nodes_detail_df[nodes_detail_df.node == node]['load_vector'].item()
        node_load = calculate_sum_list(node_load, list_load_vectors[shard - 1])

        row_index = nodes_detail_df[nodes_detail_df.node == node].index.item()

        nodes_detail_df.drop([row_index], inplace=True)

        to_append = {'node':node, 'shards': shards_list, 'load_vector': node_load}

        nodes_detail_df = nodes_detail_df.append(to_append, ignore_index=True)
        
        if(calculate_vector_module(nodes_detail_df[nodes_detail_df.node == node]['load_vector'].item()) > NWTS_module):
            # print("NWTS_module: ", NWTS_module)
            # print("load_module: ", calculate_vector_module(nodes_detail_df[nodes_detail_df.node == node]['load_vector'].item()))
            list_inactive_nodes.append(node)
            print(nodes_detail_df)
        # print(list_inactive_nodes)
        i = i + 1
    print(nodes_detail_df)
def calculate_vector_module(row):
    sum = 0
    for current_value in range(len(row)):
        sum = sum + row[current_value] ** 2
    return math.sqrt(sum)

def calculate_node_for_shard(NWTS, WSJ, WJ, list_inactive_nodes):
    deltas_j = []
    # print("NWTS: ", NWTS)
    # print("WSJ: ", WSJ)
    # print("WJ: ", WJ)
    # print("Inactive: ", list_inactive_nodes)
    for node in range(num_of_nodes):
        if(node + 1 not in list_inactive_nodes):
            deltas_j.append([node + 1, calculate_delta_j(NWTS, WSJ[node], WJ)])
            
    # print(deltas_j)
    deltas_j_df = pd.DataFrame(deltas_j, columns=['node', 'delta_j'])

    return deltas_j_df[deltas_j_df.delta_j == deltas_j_df.delta_j.max()]['node'].head(1).item()
def calculate_delta_j(NWTS, WSJ, WJ):
    first_vector_module = calculate_vector_module(calculate_diff_list(WSJ, NWTS))
    second_vector_module = calculate_vector_module(calculate_diff_list(calculate_sum_list(WSJ, WJ), NWTS))

    return first_vector_module - second_vector_module

def calculate_sum_list(list1, list2):
    sum = []

    zip_object = zip(list1, list2)

    for list1_i, list2_i in zip_object:
        sum.append(list1_i + list2_i)
    
    return sum

def calculate_diff_list(list1, list2):
    difference = []

    zip_object = zip(list1, list2)

    for list1_i, list2_i in zip_object:
        difference.append(list1_i - list2_i)
    
    return difference

if __name__ == "__main__":
    num_of_shards = int(sys.argv[1])
    num_of_nodes = int(sys.argv[2])
    algorithm = str(sys.argv[3])

    shard_allocator()
