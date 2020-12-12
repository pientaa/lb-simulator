import math
import os
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generator.generator import generator
from simulator.shard_allocator import calculate_manhattan_vector_module
from simulator.shard_allocator import diff_list
from simulator.shard_allocator import shard_allocator
from simulator.simulator import simulator

num_of_shards = 0
CLOUD_LOAD_LEVEL = "cloud_load_lvl"
LOAD_RATIO = "load_ratio"
SHARDS_PER_NODE_RATIO = "shards_per_node_ratio"


class ExperimentExecutor:
    def __init__(self):
        self.num_of_shards = 0
        self.num_of_samples = 100
        self.period = 5.0
        self.shape = 2.0
        self.scale = 0
        self.parallel_requests = 5
        self.num_of_nodes = 0
        self.experiments = [1, 2, 3]
        self.algorithms = ["random", "sequential", "SALP"]
        self.load_vectors = []

    def manualConfig(self):
        self.experiments = [int(input("Which experiment?:"))]
        self.algorithms = [str(input("Which allocation algorithm? (random/sequential/SALP):"))]

    def add_load_vectors(self, load_vectors):
        self.load_vectors = load_vectors

    def experiment_one(self):
        load_vectors_df = pd.DataFrame(self.load_vectors)
        processing_time = sum(load_vectors_df.sum(axis=1))
        periods_in_vector = load_vectors_df.shape[1]

        min_parallel_requests = round(processing_time / (periods_in_vector * self.num_of_nodes * 0.9))
        max_parallel_requests = round(processing_time / (periods_in_vector * self.num_of_nodes * 0.1))

        delays_df = pd.DataFrame(columns=['algorithm', 'nodes', 'cloud_load_lvl', 'sum_of_delay', 'delay_percentage'])
        imbalance_df = pd.DataFrame(columns=['algorithm', 'nodes', 'cloud_load_lvl', 'sum_of_imbalance', 'imbalance_percentage'])

        for parallel_requests in range(min_parallel_requests, max_parallel_requests + 1, 1):
            for algorithm in self.algorithms:
                cloud_load_lvl = processing_time / (periods_in_vector * self.num_of_nodes * parallel_requests)

                nodes_detail_df = shard_allocation(self.num_of_nodes, algorithm)
                imbalance_df = imbalance_df.append(calculate_imbalance_level(algorithm, self.num_of_nodes, load_vectors, nodes_detail_df,
                                                                             cloud_load_lvl=cloud_load_lvl), ignore_index=True)

                requests_completed_df = simulation(parallel_requests, self.period, self.num_of_nodes, algorithm)

                delays_df = delays_df.append(calculate_delays(self.num_of_samples, self.period, algorithm, self.num_of_nodes, requests_completed_df,
                                                              cloud_load_lvl=cloud_load_lvl), ignore_index=True)

        delays_df.to_csv('./experiments/' + CLOUD_LOAD_LEVEL + '/delays_' + getCurrentDateTime() + '.csv', index=False)
        imbalance_df.to_csv('./experiments/' + CLOUD_LOAD_LEVEL + '/imbalance_' + getCurrentDateTime() + '.csv', index=False)

        generate_plots(imbalance_df, delays_df, CLOUD_LOAD_LEVEL)


def experiment_executor():
    experiment = str(input("Which experiment?:"))
    while experiment not in ['1', '2', '3', 'all']:
        experiment = str(input("Which experiment? (Enter number from 1 to 3):"))

    algorithms = str(input("Which allocation algorithm?:"))
    while algorithms not in ["random", "sequential", "SALP", "all"]:
        algorithms = str(input("Which allocation algorithm? (random/sequential/SALP):"))

    if algorithms == "all":
        algorithms = ["random", "sequential", "SALP"]
    else:
        algorithms = [algorithms]

    global num_of_shards
    num_of_shards = int(input("Num of shards:"))
    num_of_samples = 100
    period = 5.0
    shape = 2.0
    scale = num_of_shards / 16.0
    parallel_requests = 5
    num_of_nodes = round(num_of_shards / 25)

    clear_directory()

    requests, load_vectors = generate_load_vectors(num_of_samples, period, shape, scale)

    ExperimentExecutor()

    if experiment == '1':
        experiment_one(num_of_samples, period, num_of_nodes, algorithms, shape, scale, load_vectors)
    elif experiment == '2':
        experiment_two(num_of_samples, period, algorithms, num_of_nodes, parallel_requests)
    elif experiment == '3':
        experiment_three(algorithms, parallel_requests, period, num_of_samples, shape, scale, load_vectors)
    elif experiment == 'all':
        experiment_one(num_of_samples, period, num_of_nodes, algorithms, shape, scale, load_vectors)
        experiment_three(algorithms, parallel_requests, period, num_of_samples, shape, scale, load_vectors)
        experiment_two(num_of_samples, period, algorithms, num_of_nodes, parallel_requests)
    else:
        print("Wrong experiment!")


def experiment_runner(x):
    return {
        '1': experiment_one(num_of_samples, period, num_of_nodes, algorithms, shape, scale, load_vectors),
        '2': experiment_two(num_of_samples, period, algorithms, num_of_nodes, parallel_requests),
        '3': experiment_three(algorithms, parallel_requests, period, num_of_samples, shape, scale, load_vectors)
    }


def generate_load_vectors(num_of_samples, period, shape, scale):
    requests, load_vectors = generator(num_of_shards, num_of_samples, period, shape, scale)

    requests.to_csv('./experiments/requests.csv')
    requests.to_csv('./generator/requests.csv')

    for vector in load_vectors:
        save_load_vector(vector)

    return requests, load_vectors


def experiment_one(num_of_samples, period, num_of_nodes, algorithms, shape, scale, load_vectors):
    load_vectors_df = pd.DataFrame(load_vectors)
    processing_time = sum(load_vectors_df.sum(axis=1))
    periods_in_vector = load_vectors_df.shape[1]

    min_parallel_requests = round(processing_time / (periods_in_vector * num_of_nodes * 0.9))
    max_parallel_requests = round(processing_time / (periods_in_vector * num_of_nodes * 0.1))

    delays_df = pd.DataFrame(columns=['algorithm', 'nodes', 'cloud_load_lvl', 'sum_of_delay', 'delay_percentage'])
    imbalance_df = pd.DataFrame(columns=['algorithm', 'nodes', 'cloud_load_lvl', 'sum_of_imbalance', 'imbalance_percentage'])

    for parallel_requests in range(min_parallel_requests, max_parallel_requests + 1, 1):
        for algorithm in algorithms:
            cloud_load_lvl = processing_time / (periods_in_vector * num_of_nodes * parallel_requests)

            nodes_detail_df = shard_allocation(num_of_nodes, algorithm)
            imbalance_df = imbalance_df.append(calculate_imbalance_level(algorithm, num_of_nodes, load_vectors, nodes_detail_df,
                                                                         cloud_load_lvl=cloud_load_lvl), ignore_index=True)

            requests_completed_df = simulation(parallel_requests, period, num_of_nodes, algorithm)

            delays_df = delays_df.append(calculate_delays(num_of_samples, period, algorithm, num_of_nodes, requests_completed_df,
                                                          cloud_load_lvl=cloud_load_lvl), ignore_index=True)

    delays_df.to_csv('./experiments/' + CLOUD_LOAD_LEVEL + '/delays_' + getCurrentDateTime() + '.csv', index=False)
    imbalance_df.to_csv('./experiments/' + CLOUD_LOAD_LEVEL + '/imbalance_' + getCurrentDateTime() + '.csv', index=False)

    generate_plots(imbalance_df, delays_df, CLOUD_LOAD_LEVEL)


def experiment_two(num_of_samples, period, algorithms, nodes, parallel_requests):
    shape = 25.0
    scale = 2.0

    mean = shape * scale

    delays_df = pd.DataFrame(columns=['algorithm', 'nodes', 'load_ratio', 'sum_of_delay', 'delay_percentage'])
    imbalance_df = pd.DataFrame(columns=['algorithm', 'nodes', 'load_ratio', 'sum_of_imbalance', 'imbalance_percentage'])

    for alfa in np.arange(1.0, shape, 1.0):
        alfa = round(alfa, 1)
        beta = mean / alfa

        clear_directory()
        requests, load_vectors = generate_load_vectors(num_of_samples, period, alfa, beta)

        load_ratio = (math.sqrt(alfa) * beta) / mean

        for algorithm in algorithms:
            nodes_detail_df = shard_allocation(nodes, algorithm)
            imbalance_df = imbalance_df.append(calculate_imbalance_level(algorithm, nodes, load_vectors, nodes_detail_df,
                                                                         load_ratio=load_ratio), ignore_index=True)

            requests_completed_df = simulation(parallel_requests, period, nodes, algorithm)

            delays_df = delays_df.append(calculate_delays(num_of_samples, period, algorithm, nodes, requests_completed_df,
                                                          load_ratio=load_ratio), ignore_index=True)

    delays_df.to_csv('./experiments/' + CLOUD_LOAD_LEVEL + '/delays_' + getCurrentDateTime() + '.csv', index=False)
    imbalance_df.to_csv('./experiments/' + CLOUD_LOAD_LEVEL + '/imbalance_' + getCurrentDateTime() + '.csv', index=False)

    generate_plots(imbalance_df, delays_df, LOAD_RATIO)


def experiment_three(algorithms, parallel_requests, period, num_of_samples, shape, scale, load_vectors):
    min_num_of_nodes = round(num_of_shards / 100)
    if (min_num_of_nodes < 1):
        min_num_of_nodes = 1
    max_num_of_nodes = round(num_of_shards / 10)

    delays_df = pd.DataFrame(columns=['algorithm', 'nodes', 'sum_of_delay', 'delay_percentage', 'shards_per_node_ratio'])

    imbalance_df = pd.DataFrame(columns=['algorithm', 'nodes', 'sum_of_imbalance', 'imbalance_percentage', 'shards_per_node_ratio'])

    for nodes in range(min_num_of_nodes, max_num_of_nodes + 1, 1):
        for algorithm in algorithms:
            shards_per_node_ratio = 1 / nodes
            nodes_detail_df = shard_allocation(nodes, algorithm)
            imbalance_df = imbalance_df.append(calculate_imbalance_level(algorithm, nodes, load_vectors, nodes_detail_df,
                                                                         shards_per_node_ratio=shards_per_node_ratio), ignore_index=True)

            requests_completed_df = simulation(parallel_requests, period, nodes, algorithm)

            delays_df = delays_df.append(calculate_delays(num_of_samples, period, algorithm, nodes, requests_completed_df,
                                                          shards_per_node_ratio=shards_per_node_ratio), ignore_index=True)

    delays_df.to_csv('./experiments/' + CLOUD_LOAD_LEVEL + '/delays_' + getCurrentDateTime() + '.csv', index=False)
    imbalance_df.to_csv('./experiments/' + CLOUD_LOAD_LEVEL + '/imbalance_' + getCurrentDateTime() + '.csv', index=False)

    generate_plots(imbalance_df, delays_df, SHARDS_PER_NODE_RATIO)


def calculate_delays(num_of_samples, period, algorithm, nodes, requests_completed_df, shards_per_node_ratio=None, cloud_load_lvl=None,
                     load_ratio=None):
    complete_processing_time = num_of_samples * period

    ignored_requests = requests_completed_df[requests_completed_df['timestamp'] >= complete_processing_time]
    observed_requests = requests_completed_df[requests_completed_df['timestamp'] < complete_processing_time]

    for index, row in observed_requests[observed_requests['actual_end_time'] > complete_processing_time].iterrows():
        observed_requests.at[index, 'actual_end_time'] = complete_processing_time
        new_delay = complete_processing_time - observed_requests[observed_requests.index == index]['expected_end_time'].item()

        if new_delay >= 0:
            observed_requests.at[index, 'delay'] = new_delay
        else:
            observed_requests.at[index, 'delay'] = 0

        total_delay = observed_requests['delay'].sum()
        percentage_delay = (total_delay / complete_processing_time) * 100.0

    if cloud_load_lvl is None and load_ratio is None and shards_per_node_ratio is not None:
        to_append = {'algorithm': algorithm, 'nodes': nodes, 'sum_of_delay': total_delay, 'delay_percentage': percentage_delay,
                     'shards_per_node_ratio': shards_per_node_ratio}
    elif cloud_load_lvl is not None and load_ratio is None and shards_per_node_ratio is None:
        to_append = {'algorithm': algorithm, 'nodes': nodes, 'cloud_load_lvl': cloud_load_lvl, 'sum_of_delay': total_delay,
                     'delay_percentage': percentage_delay}
    elif cloud_load_lvl is None and load_ratio is not None and shards_per_node_ratio is None:
        to_append = {'algorithm': algorithm, 'nodes': nodes, 'load_ratio': load_ratio, 'sum_of_delay': total_delay,
                     'delay_percentage': percentage_delay}
    else:
        to_append = None
    return to_append


def calculate_imbalance_level(algorithm, nodes, load_vectors, nodes_detail_df, shards_per_node_ratio=None, cloud_load_lvl=None, load_ratio=None):
    load_vectors_df = pd.DataFrame(load_vectors)
    WTS = load_vectors_df.sum(axis=0)
    NWTS = WTS / nodes
    NWTS_module = calculate_manhattan_vector_module(NWTS)

    sum_imbalance = 0
    for index, row in nodes_detail_df.iterrows():
        sum_imbalance = sum_imbalance + abs(
            calculate_manhattan_vector_module(diff_list(row['load_vector'], NWTS)))
    imb_lvl = (sum_imbalance / NWTS_module) * 100.0

    if cloud_load_lvl is None and load_ratio is None and shards_per_node_ratio is not None:
        to_append = {'algorithm': algorithm, 'nodes': nodes, 'sum_of_imbalance': sum_imbalance,
                     'imbalance_percentage': imb_lvl, 'shards_per_node_ratio': shards_per_node_ratio}
    elif cloud_load_lvl is not None and load_ratio is None and shards_per_node_ratio is None:
        to_append = {'algorithm': algorithm, 'nodes': nodes, 'cloud_load_lvl': cloud_load_lvl, 'sum_of_imbalance': sum_imbalance,
                     'imbalance_percentage': imb_lvl}
    elif cloud_load_lvl is None and load_ratio is not None and shards_per_node_ratio is None:
        to_append = {'algorithm': algorithm, 'nodes': nodes, 'load_ratio': load_ratio, 'sum_of_imbalance': sum_imbalance,
                     'imbalance_percentage': imb_lvl}
    else:
        to_append = None

    return to_append


def generate_plots(imbalance_lvl, delays_df, experiment):
    plt.clf()
    print(experiment)

    for group in imbalance_lvl['algorithm'].unique():
        x = imbalance_lvl[imbalance_lvl['algorithm'] == group][experiment].tolist()
        y = imbalance_lvl[imbalance_lvl['algorithm'] == group]['imbalance_percentage'].tolist()
        plt.plot(x, y, label=group, linewidth=2)

    path = experiment_dictionary().get(experiment) + "/imbalance_" + experiment + "_" + getCurrentDateTime()
    plt.legend(loc="upper right")
    # TODO: Change label to more readable representation
    plt.xlabel(experiment)
    plt.ylabel("Percentage value of imbalance")
    plt.savefig(path + ".png")

    plt.clf()
    for group in delays_df['algorithm'].unique():
        x = delays_df[delays_df['algorithm'] == group][experiment].tolist()
        y = delays_df[delays_df['algorithm'] == group]['delay_percentage'].tolist()
        plt.plot(x, y, label=group, linewidth=2)

    path = experiment_dictionary().get(experiment) + "/delays" + experiment + "_" + getCurrentDateTime()
    plt.legend(loc="upper right")
    # TODO: Change label to more readable representation
    plt.xlabel(experiment)
    plt.ylabel("Percentage value of total delay")
    plt.savefig(path + ".png")


def simulation(parallel_requests, period, nodes, algorithm):
    requests_completed_df = simulator(parallel_requests, period)
    requests_completed_df.to_csv('./experiments/' + algorithm + '/requests_completed_' + str(nodes) + '.csv',
                                 index=False)
    return requests_completed_df


def shard_allocation(nodes, algorithm):
    shard_allocated_df = shard_allocator(num_of_shards, nodes, algorithm)

    shards_allocated = []
    for index, row in shard_allocated_df.iterrows():
        node = row['node']
        for shard in row['shards']:
            shards_allocated.append([node, shard])

    shards_allocated_to_csv = pd.DataFrame(shards_allocated, columns=['node', 'shard'])

    shards_allocated_to_csv.to_csv('./experiments/' + algorithm + '/shard_allocated_' + str(nodes) + '.csv',
                                   index=False)
    shards_allocated_to_csv.to_csv('./simulator/shard_allocated.csv', index=False)

    return shard_allocated_df


def save_load_vector(load_vector):
    load_vector_file = open("./experiments/load_vectors.csv", "a")
    load_vector_string = ','.join(map(str, load_vector)) + "\n"
    load_vector_file.write(load_vector_string)
    load_vector_file.close()

    load_vector_file = open("./generator/load_vectors.csv", "a")
    load_vector_string = ','.join(map(str, load_vector)) + "\n"
    load_vector_file.write(load_vector_string)
    load_vector_file.close()


def clear_directory():
    try:
        if os.path.exists("generator/load_vectors.csv"):
            os.unlink("generator/load_vectors.csv")
        if os.path.exists("experiments/load_vectors.csv"):
            os.unlink("experiments/load_vectors.csv")
    except OSError:
        if os.path.exists("./generator/load_vectors.csv"):
            os.system("rm -f ./generator/load_vectors.csv")
        if os.path.exists("./experiments/load_vectors.csv"):
            os.system("rm -f ./experiments/load_vectors.csv")


def reset_directory():
    if os.path.exists("experiments"):
        shutil.rmtree("experiments")
    os.mkdir("experiments")
    os.mkdir("experiments/SALP")
    os.mkdir("experiments/random")
    os.mkdir("experiments/sequential")
    os.mkdir("experiments/cloud_load_lvl")


#     TODO: rest of experiments paths


def experiment_dictionary():
    return {
        CLOUD_LOAD_LEVEL: 'experiments/' + CLOUD_LOAD_LEVEL,
        LOAD_RATIO: 'experiments/' + LOAD_RATIO,
        SHARDS_PER_NODE_RATIO: 'experiments/experiment_3' + SHARDS_PER_NODE_RATIO
    }


def getCurrentDateTime():
    currentDateTime = datetime.now()

    return str(currentDateTime.year) + '-' + str(currentDateTime.month) + '-' + \
           str(currentDateTime.day) + '_' + str(currentDateTime.hour) + '-' + \
           str(currentDateTime.minute)


if __name__ == "__main__":
    reset_directory()
    experiment_executor()
