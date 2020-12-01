import os

import matplotlib.pyplot as plt
import pandas as pd

from generator.generator import generator
from simulator.shard_allocator import calculate_manhattan_vector_module
from simulator.shard_allocator import diff_list
from simulator.shard_allocator import shard_allocator
from simulator.simulator import simulator
import math

num_of_shards = 0


def experiment_executor():
    clear_directory()
    experiment = int(input("Which experiment?:"))
    while experiment not in [1, 2, 3]:
        experiment = int(input("Which experiment? (Enter number from 1 to 3):"))

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
    max_parallel_requests = 2 * parallel_requests

    num_of_nodes = round(num_of_shards / 25)
    max_num_of_nodes = round(num_of_shards / 10) + 1

    clear_directory()
    

    if(experiment == 1):
        experiment_one(num_of_samples, period, num_of_nodes, max_parallel_requests, algorithms, parallel_requests)
    elif(experiment == 2):
        experiment_two(num_of_samples, period, algorithms, shape, scale, num_of_nodes, parallel_requests)
    elif(experiment == 3):
        experiment_three(num_of_nodes, max_num_of_nodes, algorithms, parallel_requests, period, num_of_samples)
    else:
        print("Nie ten eksperyment!")



def generate_load_vectors(num_of_shards, num_of_samples, period, shape = 2.0, scale = num_of_shards / 16.0):
    requests, load_vectors = generator(num_of_shards, num_of_samples, period, shape, scale)

    requests.to_csv('./experiments/requests.csv')
    requests.to_csv('./generator/requests.csv')


    for vector in load_vectors:
        save_load_vector(vector)

    return requests, load_vectors

def experiment_one(num_of_samples, period, num_of_nodes, max_parallel_requests, algorithms, parallel_requests):
    requests, load_vectors = generate_load_vectors(num_of_shards, num_of_samples, period)

    load_vectors_df = pd.DataFrame(load_vectors)
    processing_time = sum(load_vectors_df.sum(axis = 1))
    periods_in_vector = load_vectors_df.shape[1]

    delays_df = pd.DataFrame(columns=['algorithm', 'nodes', 'num_of_parallel_requests', 'sum_of_delay', 'delay_percentage'])

    imbalance_df = pd.DataFrame(columns=['algorithm', 'nodes', 'num_of_parallel_requests', 'sum_of_imbalance', 'imbalance_percentage'])

    for cloud_efficiency in range(parallel_requests, max_parallel_requests, 1):
        for algorithm in algorithms:
            nodes_detail_df = shard_allocation(num_of_nodes, algorithm)
            imbalance_df = imbalance_df.append(calculate_imbalance_level(algorithm, num_of_nodes, load_vectors, nodes_detail_df,
                                             num_of_parallel_requests = cloud_efficiency), ignore_index=True)

            requests_completed_df = simulation(cloud_efficiency, period, num_of_nodes, algorithm)

            delays_df = delays_df.append(calculate_delays(num_of_samples, period, algorithm, num_of_nodes, requests_completed_df,
                                        num_of_parallel_requests = cloud_efficiency), ignore_index=True)
    
    generate_delays_plots(delays_df, "cloud_load", "Cloud load level", "delays_cload_load", processing_time, periods_in_vector, num_of_nodes)
    generate_imbalance_plots(imbalance_df, "cloud_load", "Cloud load level", "imbalance_lvl_cload_load", processing_time, periods_in_vector, num_of_nodes)

def experiment_two(num_of_samples, period, algorithms, shape, scale, nodes, parallel_requests):
    mean = shape * scale

    delays_df = pd.DataFrame(columns=['algorithm', 'nodes', 'load_ratio', 'sum_of_delay', 'delay_percentage'])

    imbalance_df = pd.DataFrame(columns=['algorithm', 'nodes', 'load_ratio', 'sum_of_imbalance', 'imbalance_percentage'])

    for alfa in range(int(shape), 8, 1):
        beta = mean / alfa
        clear_directory()
        requests, load_vectors = generate_load_vectors(num_of_shards, num_of_samples, period, alfa, beta)

        load_ratio = mean / (math.sqrt(alfa) * beta)

        for algorithm in algorithms:
            nodes_detail_df = shard_allocation(nodes, algorithm)
            imbalance_df = imbalance_df.append(calculate_imbalance_level(algorithm, nodes, load_vectors, nodes_detail_df,
                                            load_ratio = load_ratio), ignore_index=True)

            requests_completed_df = simulation(parallel_requests, period, nodes, algorithm)

            delays_df = delays_df.append(calculate_delays(num_of_samples, period, algorithm, nodes, requests_completed_df,
                                        load_ratio = load_ratio), ignore_index=True)

    generate_delays_plots(delays_df, "load_ratio", "Load Ratio", "delays_load_ratio")
    generate_imbalance_plots(imbalance_df, "load_ratio", "Load Ratio", "imbalance_lvl_load_ratio")
    

def experiment_three(num_of_nodes, max_num_of_nodes, algorithms, parallel_requests, period, num_of_samples):

    requests, load_vectors = generate_load_vectors(num_of_shards, num_of_samples, period)

    delays_df = pd.DataFrame(columns=['algorithm', 'nodes', 'sum_of_delay', 'delay_percentage'])

    imbalance_df = pd.DataFrame(columns=['algorithm', 'nodes', 'sum_of_imbalance', 'imbalance_percentage'])

    for nodes in range(num_of_nodes, max_num_of_nodes, 1):
        for algorithm in algorithms:
            nodes_detail_df = shard_allocation(nodes, algorithm)
            imbalance_df = imbalance_df.append(calculate_imbalance_level(algorithm, nodes, load_vectors, nodes_detail_df), ignore_index=True)

            requests_completed_df = simulation(parallel_requests, period, nodes, algorithm)

            delays_df = delays_df.append(calculate_delays(num_of_samples, period, algorithm, nodes, requests_completed_df), ignore_index=True)

    generate_delays_plots(delays_df, "shards_per_node", "Shards per node ratio", "delays_shards_per_node")
    generate_imbalance_plots(imbalance_df, "shards_per_node", "Shards per node ratio", "imbalance_lvl_shards_per_node")

def calculate_delays(num_of_samples, period, algorithm, nodes, requests_completed_df, num_of_parallel_requests = None, load_ratio = None):
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
    
    if (num_of_parallel_requests == None and load_ratio == None):
        to_append = {'algorithm': algorithm, 'nodes': nodes, 'sum_of_delay': total_delay, 'delay_percentage': percentage_delay}
    elif(num_of_parallel_requests != None and load_ratio == None):
        to_append = {'algorithm': algorithm, 'nodes': nodes, 'num_of_parallel_requests' : num_of_parallel_requests, 'sum_of_delay': total_delay,
                 'delay_percentage': percentage_delay}
    elif(num_of_parallel_requests == None and load_ratio != None):
        to_append = {'algorithm': algorithm, 'nodes': nodes, 'load_ratio' : load_ratio, 'sum_of_delay': total_delay,
                 'delay_percentage': percentage_delay}
    else:
        to_append = None
    return to_append

def calculate_imbalance_level(algorithm, nodes, load_vectors, nodes_detail_df, num_of_parallel_requests = None, load_ratio = None):
    load_vectors_df = pd.DataFrame(load_vectors)
    WTS = load_vectors_df.sum(axis=0)
    NWTS = WTS / nodes
    NWTS_module = calculate_manhattan_vector_module(NWTS)

    sum_imbalance = 0
    for index, row in nodes_detail_df.iterrows():
        sum_imbalance = sum_imbalance + abs(
            calculate_manhattan_vector_module(diff_list(row['load_vector'], NWTS)))
    imb_lvl = (sum_imbalance / NWTS_module) * 100.0

    if(num_of_parallel_requests == None and load_ratio == None):
        to_append = {'algorithm': algorithm, 'nodes': nodes, 'sum_of_imbalance': sum_imbalance,
            'imbalance_percentage': imb_lvl}
    elif(num_of_parallel_requests != None and load_ratio == None):
        to_append = {'algorithm': algorithm, 'nodes': nodes, 'num_of_parallel_requests' : num_of_parallel_requests, 'sum_of_imbalance': sum_imbalance,
            'imbalance_percentage': imb_lvl}
    elif(num_of_parallel_requests == None and load_ratio != None):
        to_append = {'algorithm': algorithm, 'nodes': nodes, 'load_ratio' : load_ratio, 'sum_of_imbalance': sum_imbalance,
            'imbalance_percentage': imb_lvl}
    else:
        to_append = None

    return to_append

def generate_imbalance_plots(imbalance_lvl, param_x, xLabel, plotTitle, processing_time = None, periods_in_vector = None, num_of_nodes = None):
    plt.clf()
    for group in imbalance_lvl['algorithm'].unique():
        if(param_x == "cloud_load"):
            x = imbalance_lvl[imbalance_lvl['algorithm'] == group]['num_of_parallel_requests'].map(lambda x: processing_time/(periods_in_vector * num_of_nodes * x)).tolist()
        elif(param_x == "load_ratio"):
            x = imbalance_lvl[imbalance_lvl['algorithm'] == group]['load_ratio'].tolist()
        elif(param_x == "shards_per_node"):
            x = imbalance_lvl[imbalance_lvl['algorithm'] == group]['nodes'].map(lambda x: num_of_shards/x).tolist()
        plt.plot(x,
                 imbalance_lvl[imbalance_lvl['algorithm'] == group]['imbalance_percentage'].tolist(),
                 label=group,
                 linewidth=2)
    plt.legend(loc="upper right")
    plt.xlabel(xLabel)
    plt.ylabel("Percentage value of imbalance")
    plt.savefig(plotTitle + ".png")

def generate_delays_plots(delays_df, param_x, xLabel, plotTitle, processing_time = None, periods_in_vector = None, num_of_nodes = None):
    plt.clf()
    for group in delays_df['algorithm'].unique():
        if(param_x == "cloud_load"):
            x = delays_df[delays_df['algorithm'] == group]['num_of_parallel_requests'].map(lambda x: processing_time/(periods_in_vector * num_of_nodes * x)).tolist()
        elif(param_x == "load_ratio"):
            x = delays_df[delays_df['algorithm'] == group]['load_ratio'].tolist()
        elif(param_x == "shards_per_node"):
            x = delays_df[delays_df['algorithm'] == group]['nodes'].map(lambda x: num_of_shards/x).tolist()
        plt.plot(x,
                 delays_df[delays_df['algorithm'] == group]['delay_percentage'].tolist(),
                 label=group,
                 linewidth=2)
    plt.legend(loc="upper right")
    plt.xlabel(xLabel)
    plt.ylabel("Percentage value of total delay")
    plt.savefig(plotTitle + ".png")


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
        os.remove("experiments/load_vectors.csv")
        os.remove("generator/load_vectors.csv")
    except OSError:
        os.system("rm -f ./experiments/load_vectors.csv")
        os.system("rm -f ./generator/load_vectors.csv")


if __name__ == "__main__":
    experiment_executor()
