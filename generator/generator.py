import numpy as np
import pandas as pd
import sys
import random
import math
import os
import matplotlib.pyplot as plt
import scipy.special as sps  
from itertools import chain


def generator(num_of_shards, num_of_samples, period, shape, scale):

    tasks = [int(round(number, 0)) for number in np.random.gamma(shape, scale, num_of_samples)]

    timestamps = [round(number, 3) for number in generate_time_stamps(tasks, period)]

    loads = [round(number, 3) for number in np.random.gamma(shape, scale, len(timestamps))]

    shards = np.random.randint(1, num_of_shards + 1, len(timestamps))

    requests = pd.DataFrame(list(zip(timestamps, shards, loads)), columns=['timestamp', 'shard', 'load'])

# Plot density
    # count, bins, ignored = plt.hist(tasks, 25, density=True)
    # y = bins**(shape-1)*(np.exp(-bins/scale) /  
    #                  (sps.gamma(shape)*scale**shape))
    # plt.plot(bins, y, linewidth=2, color='r')  
    # plt.show()


    requests.to_csv('./generator/requests.csv') 

    generate_load_vectors(requests, period, num_of_shards)

def generate_time_stamps(tasks, period):
    timestamps = []
    for i in range(len(tasks)):
        random_t = np.random.gamma(1.0, 1.0, tasks[i])
        scaled_t = [round(float(number), 3) * period + float(period) * i for number in normalize(random_t)]
        sorted_t = sorted(scaled_t, key=float)
        timestamps.append(sorted_t)
    
    return flatten(timestamps)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def flatten(listOfLists):
    return list(chain.from_iterable(listOfLists))

def generate_load_vectors(requests, period, num_of_shards):
    load_vectors = []
    max_vector_size = 0

    for (shard, group) in requests.groupby('shard'):
        load_vector = [0.0] * num_of_samples

        for current_period_index in range(len(load_vector)):
            current_requests = group[(group['timestamp'] >= period * current_period_index) & (group['timestamp'] < period * (current_period_index + 1))]

            if(not current_requests.empty):
                for index, current_request in current_requests.iterrows():
                    load_vector = calculate_load_vector(current_request, current_period_index, load_vector)

        if (max_vector_size < len(load_vector)):
            max_vector_size = len(load_vector)

        load_vectors.append(load_vector)

    for vector in load_vectors:
        while (len(vector) < max_vector_size):
            vector.append(0.0)
        
        save_load_vector(shard, vector)

def calculate_load_vector(current_request, current_period_index, load_vector):
    start_time = float(current_request['timestamp'])
    end_time = round(start_time + float(current_request['load']), 3)
    last_period_index = int(math.floor(end_time / period))
    current_load = float(current_request['load']) / float(period)

    num_of_full_periods = last_period_index - current_period_index - 1

    if (num_of_full_periods == -1):
        load_vector[current_period_index] = load_vector[current_period_index] + current_load

    if (num_of_full_periods == 0):
        first_period_load = (((current_period_index + 1) * period - start_time) / float(period))
        second_period_load = current_load - first_period_load

        load_vector[current_period_index] = load_vector[current_period_index] + first_period_load

        if (len(load_vector) > current_period_index + 1):
            load_vector[current_period_index + 1] = load_vector[current_period_index + 1] + second_period_load
        else:
            load_vector.append(second_period_load)

    if (num_of_full_periods > 0):
        first_period_load = (((current_period_index + 1) * period - start_time) / float(period)) 
        last_period_load = current_load - first_period_load - num_of_full_periods

        last_period_index = current_period_index + num_of_full_periods + 2

        load_vector[current_period_index] = load_vector[current_period_index] + first_period_load

        if (len(load_vector) > last_period_index):                
            load_vector[last_period_index] = load_vector[last_period_index] + last_period_load
        else:
            load_vector.append(last_period_load)

        for j in range(num_of_full_periods):
            load_vector[current_period_index + j + 1] = load_vector[current_period_index + j + 1] + 1
            
    return [round(load, 3) for load in load_vector]

def save_load_vector(shard, load_vector):
    load_vector_file = open("./generator/load_vectors.csv", "a")
    load_vector_string = ','.join(map(str, load_vector)) + "\n"
    load_vector_file.write(load_vector_string)
    load_vector_file.close()

def clear_directory():
    try:
        os.remove("generator/requests.csv")
        os.remove("generator/load_vectors.csv")
    except OSError:
        os.system("rm -f ./generator/load_vectors.csv")
        os.system("rm -f ./generator/requests.csv")

if __name__ == "__main__":
    period = float(sys.argv[3])
    num_of_samples = int(sys.argv[2])

    clear_directory()

    generator(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))
