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

    # print(tasks)

    timestamps = [round(number, 3) for number in generate_time_stamps(tasks, period)]

# We can parametrize this distribution in the future, too
    loads = [round(number, 3) for number in np.random.gamma(2.0, 2.0, len(timestamps))]

    shards = np.random.randint(1, num_of_shards + 1, len(timestamps))

    requests = pd.DataFrame(list(zip(timestamps, shards, loads)), columns=['timestamp', 'shard', 'load'])

# Test if correct number of requests generated
    sum = 0

    for task in tasks:
        sum += task

    assert len(requests) == sum

    requests.to_csv('./generator/requests.csv') 

# Plot density
    count, bins, ignored = plt.hist(tasks, 25, density=True)
    y = bins**(shape-1)*(np.exp(-bins/scale) /  
                     (sps.gamma(shape)*scale**shape))
    plt.plot(bins, y, linewidth=2, color='r')  
    plt.show()

    generate_load_vectors(requests, period, num_of_shards)

def generate_time_stamps(tasks, period):
    timestamps = []
    for i in range(len(tasks)):
        random_t = np.random.gamma(1.0, 1.0, tasks[i])
        scaled_t = [round(float(number), 3) * period + float(period) * i for number in normalize_vector(random_t)]
        sorted_t = sorted(scaled_t, key=float)
        timestamps.append(sorted_t)
    
    return flatten(timestamps)

def normalize_vector(v):
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
        load_vector = [0.0] * int(math.ceil(group['timestamp'].max() / period))

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

def calculate_load_vector(current_request, current_load_index, load_vector):
    start_time = float(current_request['timestamp'])
    end_time = round(start_time + float(current_request['load']), 3)

    last_period_index = int(math.floor(end_time / period))
    first_period_index = int(math.floor(start_time / period))
    num_of_periods = 1 + last_period_index - first_period_index

    current_load = round(float(current_request['load']), 3)

    if (num_of_periods == 1):
        load_vector[first_period_index] = load_vector[first_period_index] + normalize(current_load)

    if (num_of_periods == 2):
        first_period_load = (first_period_index + 1) * period - start_time
        second_period_load = current_load - first_period_load

        first_period_load = normalize(first_period_load)
        second_period_load = normalize(second_period_load)

        load_vector[first_period_index] += first_period_load

        if (len(load_vector) > first_period_index + 1):
            load_vector[first_period_index + 1] += second_period_load
        else:
            load_vector.append(second_period_load)

    if (num_of_periods > 2):
        first_period_load = (first_period_index + 1) * period - start_time
        num_of_full_periods = num_of_periods - 2

        first_period_load = normalize(first_period_load)   

        load_vector[current_load_index] = load_vector[current_load_index] + first_period_load

        for index in range(num_of_full_periods):
            current_index = first_period_index + index
            if (len(load_vector) > current_index + 1):    
                load_vector[current_index] = load_vector[current_index] + 1.0
            else:
                load_vector.append(1.0)
        
        last_period_load = normalize(current_load - first_period_load - num_of_full_periods)

        if (len(load_vector) > last_period_index):                
            load_vector[last_period_index] = load_vector[last_period_index] + last_period_load
        else:
            load_vector.append(last_period_load)
        
    return [round(load, 3) for load in load_vector]

def normalize(load):
    return round(load / float(period), 3)

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
