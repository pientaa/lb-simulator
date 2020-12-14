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
        self.experiments = []
        self.algorithms = []
        self.load_vectors = []
        self.shard_on_nodes = pd.DataFrame(columns=["shard", "node"])
        self.requests_completed = pd.DataFrame()
        self.current_algorithm = "random"
        self.delays_df = pd.DataFrame(columns=['algorithm', 'nodes', 'cloud_load_lvl', 'sum_of_delay', 'delay_percentage'])
        self.imbalance_df = pd.DataFrame(columns=['algorithm', 'nodes', 'cloud_load_lvl', 'sum_of_imbalance', 'imbalance_percentage'])

    def print(self):
        print("Shards: " + str(self.num_of_shards))
        print("Nodes: " + str(self.num_of_nodes))
        print("Period: " + str(self.period))
        print("Shape: " + str(self.shape))
        print("Experiments: " + str(self.experiments))

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

    def shard_allocation(self, algorithm):
        shard_allocated_df = shard_allocator(self.num_of_shards, self.num_of_nodes, algorithm)

        shard_allocated_df.to_csv('./experiments/' + algorithm + '/shard_allocated_' + str(self.num_of_nodes) + '.csv',
                                  index=False)
        shard_allocated_df.to_csv('./simulator/shard_allocated.csv', index=False)

        self.shard_on_nodes = shard_allocated_df
        return self

    def simulation(self, algorithm):
        self.requests_completed = simulator(self.parallel_requests, self.period)
        path = './experiments/' + algorithm + '/requests_completed_' + str(self.num_of_nodes) + '.csv'
        self.requests_completed.to_csv(path, index=False)
        return self

    def calculate_imbalance_level(self, algorithm, experiment, experiment_value):
        load_vectors_df = pd.DataFrame(self.load_vectors)

        WTS = load_vectors_df.sum(axis=0)
        NWTS = WTS / self.num_of_nodes
        NWTS_module = calculate_manhattan_vector_module(NWTS)

        sum_imbalance = 0
        for (node, group) in self.shard_on_nodes.groupby('node'):
            vectors = [int]
            for shard in group["shard"].to_list():
                vectors.append(self.load_vectors[shard - 1])
            node_load_vector = pd.DataFrame(vectors).sum(axis=0)

            sum_imbalance = sum_imbalance + abs(
                calculate_manhattan_vector_module(diff_list(node_load_vector, NWTS)))

        imb_lvl = (sum_imbalance / NWTS_module) * 100.0

        new_row = {'algorithm': algorithm, 'nodes': self.num_of_nodes, 'sum_of_imbalance': sum_imbalance, 'imbalance_percentage': imb_lvl,
                   experiment: experiment_value}

        self.imbalance_df = self.imbalance_df.append(new_row, ignore_index=True)

        return self

    def run_experiments(self):
        for experiment in self.experiments:
            if experiment == '1':
                self.experiment_one()
            if experiment == '2':
                self.experiment_two()
            if experiment == '3':
                self.experiment_three()

    def experiment_one(self):
        self.clear()
        load_vectors_df = pd.DataFrame(self.load_vectors)
        processing_time = sum(load_vectors_df.sum(axis=1))
        periods_in_vector = load_vectors_df.shape[1]

        min_parallel_requests = round(processing_time / (periods_in_vector * self.num_of_nodes * 0.9))
        max_parallel_requests = round(processing_time / (periods_in_vector * self.num_of_nodes * 0.1))

        for parallel_requests in range(min_parallel_requests, max_parallel_requests + 1, 1):
            self.parallel_requests = parallel_requests

            for algorithm in self.algorithms:
                cloud_load_lvl = processing_time / (periods_in_vector * self.num_of_nodes * parallel_requests)

                self.shard_allocation(algorithm). \
                    calculate_imbalance_level(algorithm, CLOUD_LOAD_LEVEL, cloud_load_lvl). \
                    simulation(algorithm). \
                    calculate_delays(algorithm, CLOUD_LOAD_LEVEL, cloud_load_lvl)

        self.save_delays_and_imbalance(CLOUD_LOAD_LEVEL)
        self.generatePlots(CLOUD_LOAD_LEVEL)

    def experiment_two(self):
        self.clear()
        self.shape = 25.0
        self.scale = 2.0

        mean = self.shape * self.scale

        for alfa in np.arange(1.0, self.shape, 1.0):
            self.shape = round(alfa, 1)
            self.scale = round(mean / self.shape, 3)

            requests, load_vectors = generate_load_vectors(self.num_of_shards, self.num_of_samples, self.period, self.shape, self.scale)
            self.load_vectors = load_vectors

            load_ratio = (math.sqrt(self.shape) * self.scale) / mean

            for algorithm in self.algorithms:
                self.shard_allocation(algorithm). \
                    calculate_imbalance_level(algorithm, LOAD_RATIO, load_ratio). \
                    simulation(algorithm). \
                    calculate_delays(algorithm, LOAD_RATIO, load_ratio)

        self.save_delays_and_imbalance(LOAD_RATIO)
        self.generatePlots(LOAD_RATIO)

    def experiment_three(self):
        self.clear()
        min_num_of_nodes = round(self.num_of_shards / 100)
        if min_num_of_nodes < 1:
            min_num_of_nodes = 1
        max_num_of_nodes = round(self.num_of_shards / 10)

        for nodes in range(min_num_of_nodes, max_num_of_nodes + 1, 1):
            self.num_of_nodes = nodes
            shards_per_node_ratio = 1 / nodes

            for algorithm in self.algorithms:
                self.shard_allocation(algorithm). \
                    calculate_imbalance_level(algorithm, SHARDS_PER_NODE_RATIO, shards_per_node_ratio). \
                    simulation(algorithm). \
                    calculate_delays(algorithm, SHARDS_PER_NODE_RATIO, shards_per_node_ratio)

        self.save_delays_and_imbalance(SHARDS_PER_NODE_RATIO)
        self.generatePlots(SHARDS_PER_NODE_RATIO)

    def calculate_delays(self, algorithm, experiment, experiment_value):
        complete_processing_time = self.num_of_samples * self.period

        observed_requests = self.requests_completed[self.requests_completed['timestamp'] < complete_processing_time]

        for index, row in observed_requests[observed_requests['actual_end_time'] > complete_processing_time].iterrows():
            observed_requests.at[index, 'actual_end_time'] = complete_processing_time
            new_delay = complete_processing_time - observed_requests[observed_requests.index == index]['expected_end_time'].item()

            if new_delay >= 0:
                observed_requests.at[index, 'delay'] = new_delay
            else:
                observed_requests.at[index, 'delay'] = 0

        total_delay = observed_requests['delay'].sum()
        percentage_delay = (total_delay / complete_processing_time) * 100.0

        new_row = {'algorithm': algorithm, 'nodes': self.num_of_nodes, 'sum_of_delay': total_delay, 'delay_percentage': percentage_delay,
                   experiment: experiment_value}

        self.delays_df = self.delays_df.append(new_row, ignore_index=True)

        return self

    def save_delays_and_imbalance(self, experiment):
        self.delays_df.to_csv('./experiments/' + experiment + '/delays_' + getCurrentDateTime() + '.csv', index=False)
        self.imbalance_df.to_csv('./experiments/' + experiment + '/imbalance_' + getCurrentDateTime() + '.csv', index=False)

    def generatePlots(self, experiment):
        plt.clf()

        for group in self.imbalance_df['algorithm'].unique():
            x = self.imbalance_df[self.imbalance_df['algorithm'] == group][experiment].tolist()
            y = self.imbalance_df[self.imbalance_df['algorithm'] == group]['imbalance_percentage'].tolist()
            plt.plot(x, y, label=group, linewidth=2)

        path = "experiments/" + experiment + "/imbalance" + "_" + getCurrentDateTime()
        plt.legend(loc="upper right")
        # TODO: Change label to more readable representation
        plt.xlabel(experiment)
        plt.ylabel("Percentage value of imbalance")
        plt.savefig(path + ".png")

        plt.clf()
        for group in self.delays_df['algorithm'].unique():
            x = self.delays_df[self.delays_df['algorithm'] == group][experiment].tolist()
            y = self.delays_df[self.delays_df['algorithm'] == group]['delay_percentage'].tolist()
            plt.plot(x, y, label=group, linewidth=2)

        path = "experiments/" + experiment + "/delays" + experiment + "_" + getCurrentDateTime()
        plt.legend(loc="upper right")
        # TODO: Change label to more readable representation
        plt.xlabel(experiment)
        plt.ylabel("Percentage value of total delay")
        plt.savefig(path + ".png")

    def clear(self):
        self.delays_df = pd.DataFrame(columns=['algorithm', 'nodes', 'cloud_load_lvl', 'sum_of_delay', 'delay_percentage'])
        self.imbalance_df = pd.DataFrame(columns=['algorithm', 'nodes', 'cloud_load_lvl', 'sum_of_imbalance', 'imbalance_percentage'])


def experiment_executor():
    clear_directory()

    executor = ExperimentExecutor(). \
        manual_config()

    executor.print()

    requests, load_vectors = generate_load_vectors(executor.num_of_shards, executor.num_of_samples, executor.period, executor.shape, executor.scale)

    executor.add_load_vectors(load_vectors). \
        run_experiments()


def generate_load_vectors(num_of_shards, num_of_samples, period, shape, scale):
    clear_directory()
    requests, load_vectors = generator(num_of_shards, num_of_samples, period, shape, scale)

    requests.to_csv('./experiments/requests.csv')
    requests.to_csv('./generator/requests.csv')

    for vector in load_vectors:
        save_load_vector(vector)

    return requests, load_vectors


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


def getCurrentDateTime():
    currentDateTime = datetime.now()

    return str(currentDateTime.year) + '-' + str(currentDateTime.month) + '-' + \
           str(currentDateTime.day) + '_' + str(currentDateTime.hour) + '-' + \
           str(currentDateTime.minute)


if __name__ == "__main__":
    reset_directory()
    experiment_executor()
