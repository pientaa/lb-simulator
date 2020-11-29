import sys

import numpy as np
import pandas as pd

num_of_shards = 0
num_of_nodes = 0
algorithm = ''


def shard_allocator(shards, nodes, algorithm_name):
    global num_of_shards, num_of_nodes, algorithm
    num_of_shards = shards
    num_of_nodes = nodes
    algorithm = algorithm_name

    print("Shard allocation started with following params:")
    print("{ shards: " + str(num_of_shards) + ", nodes: " + str(num_of_nodes) + ", algorithm: " + algorithm + "  } \n")

    if algorithm not in ["random", "sequential", "SALP"]:
        sys.exit("Pass one of allocation algorithms: random/sequential/SALP as third param.")

    if algorithm == "random":
        shards_on_nodes_df = random_allocation()

    if algorithm == "sequential":
        shards_on_nodes_df = sequential_allocation()

    if algorithm == "SALP":
        shards_on_nodes_df = SALP_allocation()

    return shards_on_nodes_df


def random_allocation():
    load_vectors_df = pd.read_csv("./generator/load_vectors.csv", header=None)
    periods_in_vector = load_vectors_df.shape[1]

    shards_on_nodes = []
    current_node = 1

    shards_shuffled = np.random.choice(range(1, num_of_shards + 1), num_of_shards, replace=False)

    for i in range(len(shards_shuffled)):
        shard = shards_shuffled[i]

        shards_on_nodes.append([shard, current_node])

        if (i + 1) % int(num_of_shards / num_of_nodes) == 0:
            if current_node != num_of_nodes:
                current_node += 1

    empty_load_vector = [0] * periods_in_vector
    nodes_detail = []
    for node in range(num_of_nodes):
        nodes_detail.append([node + 1, [], empty_load_vector])

    nodes_detail_df = pd.DataFrame(nodes_detail, columns=['node', 'shards', 'load_vector'])

    for shard in shards_on_nodes:
        current_node = shard[1]
        shards = nodes_detail_df[nodes_detail_df.node == current_node]['shards'].item()
        shards.append(shard[0])

        node_load_vector = nodes_detail_df[nodes_detail_df.node == current_node]['load_vector'].item()
        node_load_vector = sum_list(node_load_vector, load_vectors_df.loc[shard[0] - 1])

        to_append = {'node': shard[1], 'shards': shards, 'load_vector': node_load_vector}

        row_index = nodes_detail_df[nodes_detail_df.node == shard[1]].index.item()
        nodes_detail_df.drop([row_index], inplace=True)

        nodes_detail_df = nodes_detail_df.append(to_append, ignore_index=True)

    return nodes_detail_df


def sequential_allocation():
    load_vectors_df = pd.read_csv("./generator/load_vectors.csv", header=None)
    periods_in_vector = load_vectors_df.shape[1]

    shards_on_nodes = []
    current_node = 1

    for shard in range(num_of_shards):

        shards_on_nodes.append([shard + 1, current_node])

        if (shard + 1) % int(num_of_shards / num_of_nodes) == 0:
            if current_node != num_of_nodes:
                current_node += 1

    zeros_list = [0] * periods_in_vector
    nodes_detail = []
    for node in range(num_of_nodes):
        nodes_detail.append([node + 1, [], zeros_list])

    nodes_detail_df = pd.DataFrame(nodes_detail, columns=['node', 'shards', 'load_vector'])

    for shard in shards_on_nodes:
        shards_list = nodes_detail_df[nodes_detail_df.node == shard[1]]['shards'].item()
        shards_list.append(shard[0])

        load_node_vector = nodes_detail_df[nodes_detail_df.node == shard[1]]['load_vector'].item()
        load_node_vector = sum_list(load_node_vector, load_vectors_df.loc[shard[0] - 1])

        to_append = {'node': shard[1], 'shards': shards_list, 'load_vector': load_node_vector}

        row_index = nodes_detail_df[nodes_detail_df.node == shard[1]].index.item()
        nodes_detail_df.drop([row_index], inplace=True)

        nodes_detail_df = nodes_detail_df.append(to_append, ignore_index=True)

    return nodes_detail_df


def SALP_allocation():
    load_vectors_df = pd.read_csv("./generator/load_vectors.csv", header=None)
    WTS = load_vectors_df.sum(axis=0)
    periods_in_vector = load_vectors_df.shape[1]

    NWTS = WTS / num_of_nodes

    NWTS_module = calculate_manhattan_vector_module(NWTS)

    modules_list = []

    for index, row in load_vectors_df.iterrows():
        modules_list.append([calculate_manhattan_vector_module(row), index + 1, row])

    modules_sorted_df = pd.DataFrame(modules_list, columns=["module", "shard", "load_vector"]).sort_values('module',
                                                                                                           ascending=False)

    list_inactive_nodes = []

    zeros_list = [0] * periods_in_vector
    nodes_detail = []
    for node in range(num_of_nodes):
        nodes_detail.append([node + 1, [], zeros_list])

    nodes_detail_df = pd.DataFrame(nodes_detail, columns=['node', 'shards', 'load_vector'])
    for shard in modules_sorted_df['shard']:

        node = calculate_node_for_shard(NWTS, nodes_detail_df,
                                        modules_sorted_df[modules_sorted_df.shard == shard]['load_vector'].item(),
                                        list_inactive_nodes)

        shards_list = nodes_detail_df[nodes_detail_df.node == node]['shards'].item()
        shards_list.append(shard)

        node_load = nodes_detail_df[nodes_detail_df.node == node]['load_vector'].item()
        node_load = sum_list(node_load,
                             modules_sorted_df[modules_sorted_df.shard == shard]['load_vector'].item())

        row_index = nodes_detail_df[nodes_detail_df.node == node].index.item()

        nodes_detail_df.drop([row_index], inplace=True)

        to_append = {'node': node, 'shards': shards_list, 'load_vector': node_load}

        nodes_detail_df = nodes_detail_df.append(to_append, ignore_index=True)

        if calculate_manhattan_vector_module(
                nodes_detail_df[nodes_detail_df.node == node]['load_vector'].item()) > NWTS_module:
            list_inactive_nodes.append(node)

    return nodes_detail_df


def calculate_manhattan_vector_module(row):
    sum = 0
    for current_value in range(len(row)):
        sum = sum + row[current_value]
    return sum


def calculate_node_for_shard(NWTS, WSJ_df, WJ, list_inactive_nodes):
    deltas_j = []

    for node in range(num_of_nodes):
        if node + 1 not in list_inactive_nodes:
            deltas_j.append(
                [node + 1, calculate_delta_j(NWTS, WSJ_df[WSJ_df.node == node + 1]['load_vector'].item(), WJ)])

    deltas_j_df = pd.DataFrame(deltas_j, columns=['node', 'delta_j'])

    return deltas_j_df[deltas_j_df.delta_j == deltas_j_df.delta_j.max()]['node'].head(1).item()


def calculate_delta_j(NWTS, WSJ, WJ):
    first_vector_module = calculate_manhattan_vector_module(diff_list(WSJ, NWTS))
    second_vector_module = calculate_manhattan_vector_module(diff_list(sum_list(WSJ, WJ), NWTS))

    return first_vector_module - second_vector_module


def sum_list(list1, list2):
    sum = []

    zip_object = zip(list1, list2)

    for list1_i, list2_i in zip_object:
        sum.append(list1_i + list2_i)

    return sum


def diff_list(list1, list2):
    difference = []

    zip_object = zip(list1, list2)

    for list1_i, list2_i in zip_object:
        difference.append(list1_i - list2_i)

    return difference


if __name__ == "__main__":
    shard_allocator()
