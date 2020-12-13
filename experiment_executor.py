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
        self.experiments = ['1', '2', '3']
        self.algorithms = ["random", "sequential", "SALP"]
        self.load_vectors = []

    def print(self):
        print("Shards: " + str(self.num_of_shards))
        print("Nodes: " + str(self.num_of_nodes))
        print("Period: " + str(self.period))
        print("Shape: " + str(self.shape))

    def manual_config(self):
        experiment = str(input("Which experiment?:"))
        if experiment == "all":
            experiment = ["1", "2", "3"]
        else:
            experiment = [experiment]
        self.experiments = experiment

        algorithm = str(input("Which allocation algorithm? (random/sequential/SALP):"))

        if algorithm == "all":
            algorithm = ["random", "sequential", "SALP"]
        else:
            algorithm = [algorithm]
        self.algorithms = algorithm

        self.num_of_shards = int(input("Num of shards:"))
        self.num_of_samples = 100
        self.period = 5.0
        self.shape = 2.0
        self.scale = self.num_of_shards / 16.0
        self.parallel_requests = 5
        self.num_of_nodes = 1

        return self

    def add_load_vectors(self, load_vectors):
        self.load_vectors = load_vectors
        return self

    def run_experiments(self):
        for experiment in self.experiments:
            self.experiment_runner().get(experiment)

    def experiment_runner(self):
        return {
            '1': self.experiment_one(),
            '2': self.experiment_two(),
            '3': self.experiment_three()
        }

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

                nodes_detail_df = shard_allocation(self.num_of_shards, self.num_of_nodes, algorithm)
                imbalance_df = imbalance_df.append(calculate_imbalance_level(algorithm, self.num_of_nodes, self.load_vectors, nodes_detail_df,
                                                                             cloud_load_lvl=cloud_load_lvl), ignore_index=True)

                requests_completed_df = simulation(parallel_requests, self.period, self.num_of_nodes, algorithm)

                delays_df = delays_df.append(calculate_delays(self.num_of_samples, self.period, algorithm, self.num_of_nodes, requests_completed_df,
                                                              cloud_load_lvl=cloud_load_lvl), ignore_index=True)

        delays_df.to_csv('./experiments/' + CLOUD_LOAD_LEVEL + '/delays_' + getCurrentDateTime() + '.csv', index=False)
        imbalance_df.to_csv('./experiments/' + CLOUD_LOAD_LEVEL + '/imbalance_' + getCurrentDateTime() + '.csv', index=False)

        generate_plots(imbalance_df, delays_df, CLOUD_LOAD_LEVEL)

    def experiment_two(self):
        self.shape = 25.0
        self.scale = 2.0

        mean = self.shape * self.scale

        delays_df = pd.DataFrame(columns=['algorithm', 'nodes', 'load_ratio', 'sum_of_delay', 'delay_percentage'])
        imbalance_df = pd.DataFrame(columns=['algorithm', 'nodes', 'load_ratio', 'sum_of_imbalance', 'imbalance_percentage'])

        for alfa in np.arange(1.0, self.shape, 1.0):
            alfa = round(alfa, 1)
            beta = mean / alfa

            clear_directory()
            requests, load_vectors = generate_load_vectors(self.num_of_shards, self.num_of_samples, self.period, alfa, beta)

            load_ratio = (math.sqrt(alfa) * beta) / mean

            for algorithm in self.algorithms:
                nodes_detail_df = shard_allocation(self.num_of_shards, self.num_of_nodes, algorithm)
                imbalance_df = imbalance_df.append(calculate_imbalance_level(algorithm, self.num_of_nodes, load_vectors, nodes_detail_df,
                                                                             load_ratio=load_ratio), ignore_index=True)

                requests_completed_df = simulation(self.parallel_requests, self.period, self.num_of_nodes, algorithm)

                delays_df = delays_df.append(calculate_delays(self.num_of_samples, self.period, algorithm, self.num_of_nodes, requests_completed_df,
                                                              load_ratio=load_ratio), ignore_index=True)

        delays_df.to_csv('./experiments/' + CLOUD_LOAD_LEVEL + '/delays_' + getCurrentDateTime() + '.csv', index=False)
        imbalance_df.to_csv('./experiments/' + CLOUD_LOAD_LEVEL + '/imbalance_' + getCurrentDateTime() + '.csv', index=False)

        generate_plots(imbalance_df, delays_df, LOAD_RATIO)

    def experiment_three(self):
        min_num_of_nodes = round(self.num_of_shards / 100)
        if min_num_of_nodes < 1:
            min_num_of_nodes = 1
        max_num_of_nodes = round(self.num_of_shards / 10)

        delays_df = pd.DataFrame(columns=['algorithm', 'nodes', 'sum_of_delay', 'delay_percentage', 'shards_per_node_ratio'])

        imbalance_df = pd.DataFrame(columns=['algorithm', 'nodes', 'sum_of_imbalance', 'imbalance_percentage', 'shards_per_node_ratio'])

        for nodes in range(min_num_of_nodes, max_num_of_nodes + 1, 1):
            for algorithm in self.algorithms:
                shards_per_node_ratio = 1 / nodes
                nodes_detail_df = shard_allocation(nodes, algorithm)
                imbalance_df = imbalance_df.append(calculate_imbalance_level(algorithm, nodes, self.load_vectors, nodes_detail_df,
                                                                             shards_per_node_ratio=shards_per_node_ratio), ignore_index=True)

                requests_completed_df = simulation(self.parallel_requests, self.period, nodes, algorithm)

                delays_df = delays_df.append(calculate_delays(self.num_of_samples, self.period, algorithm, nodes, requests_completed_df,
                                                              shards_per_node_ratio=shards_per_node_ratio), ignore_index=True)

        delays_df.to_csv('./experiments/' + CLOUD_LOAD_LEVEL + '/delays_' + getCurrentDateTime() + '.csv', index=False)
        imbalance_df.to_csv('./experiments/' + CLOUD_LOAD_LEVEL + '/imbalance_' + getCurrentDateTime() + '.csv', index=False)

        generate_plots(imbalance_df, delays_df, SHARDS_PER_NODE_RATIO)


def experiment_executor():
    clear_directory()

    executor = ExperimentExecutor(). \
        manual_config()

    executor.print()

    requests, load_vectors = generate_load_vectors(executor.num_of_shards, executor.num_of_samples, executor.period, executor.shape, executor.scale)

    executor.add_load_vectors(load_vectors). \
        run_experiments()


def generate_load_vectors(num_of_shards, num_of_samples, period, shape, scale):
    requests, load_vectors = generator(num_of_shards, num_of_samples, period, shape, scale)

    requests.to_csv('./experiments/requests.csv')
    requests.to_csv('./generator/requests.csv')

    for vector in load_vectors:
        save_load_vector(vector)

    return requests, load_vectors


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

    for group in imbalance_lvl['algorithm'].unique():
        x = imbalance_lvl[imbalance_lvl['algorithm'] == group][experiment].tolist()
        y = imbalance_lvl[imbalance_lvl['algorithm'] == group]['imbalance_percentage'].tolist()
        plt.plot(x, y, label=group, linewidth=2)

    path = "experiments/" + experiment + "/imbalance" + "_" + getCurrentDateTime()
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

    path = "experiments/" + experiment + "/delays" + experiment + "_" + getCurrentDateTime()
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


def shard_allocation(num_of_shards, nodes, algorithm):
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
    os.mkdir("experiments/" + CLOUD_LOAD_LEVEL)
    os.mkdir("experiments/" + LOAD_RATIO)
    os.mkdir("experiments/" + SHARDS_PER_NODE_RATIO)


#     TODO: rest of experiments paths


def getCurrentDateTime():
    currentDateTime = datetime.now()

    return str(currentDateTime.year) + '-' + str(currentDateTime.month) + '-' + \
           str(currentDateTime.day) + '_' + str(currentDateTime.hour) + '-' + \
           str(currentDateTime.minute)


if __name__ == "__main__":
    reset_directory()
    experiment_executor()
