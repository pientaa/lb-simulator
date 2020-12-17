import math
import random
import sys
from itertools import chain

import numpy as np
import pandas as pd

period = 5.0


def generator(num_of_shards, num_of_samples, new_period, shape, scale):
    global period
    period = new_period

    print("Generator started with following params:")
    print("{ num_of_shards: " + str(num_of_shards) + ", num_of_samples: " + str(num_of_samples) +
          ", period: " + str(period) + ", shape: " + str(shape) + ", scale: " + str(scale) + "  } \n")

    tasks = [int(round(number, 0)) for number in np.random.gamma(shape, scale, num_of_samples)]

    timestamps = [round(number, 3) for number in generate_time_stamps(tasks)]

    # We can parametrize this distribution in the future, too
    loads = [round(number, 3) for number in np.random.gamma(2.0, 2.0, len(timestamps))]

    shards = np.random.randint(1, num_of_shards + 1, len(timestamps))

    requests = pd.DataFrame(list(zip(timestamps, shards, loads)), columns=['timestamp', 'shard', 'load'])

    # Test if correct number of requests generated
    sum = 0

    for task in tasks:
        sum += task

    assert len(requests) == sum

    return requests, generate_load_vectors(requests, num_of_shards)


def generate_time_stamps(tasks):
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


def flatten(list_of_lists):
    return list(chain.from_iterable(list_of_lists))


def generate_load_vectors(requests, num_of_shards):
    load_vectors = []
    max_vector_size = 0

    for (shard, group) in requests.groupby('shard'):
        load_vector = [0.0] * int(math.ceil(group['timestamp'].max() / period))

        for current_period_index in range(len(load_vector)):
            current_requests = group[(group['timestamp'] >= period * current_period_index) & (
                    group['timestamp'] < period * (current_period_index + 1))]

            if not current_requests.empty:
                for index, current_request in current_requests.iterrows():
                    load_vector = calculate_load_vector(current_request, current_period_index, load_vector)

        if max_vector_size < len(load_vector):
            max_vector_size = len(load_vector)

        load_vectors.append(load_vector)

    for vector in load_vectors:
        while len(vector) < max_vector_size:
            vector.append(0.0)

    while len(load_vectors) < num_of_shards:
        load_vectors.append([0] * max_vector_size)

    random.shuffle(load_vectors)

    return load_vectors


def calculate_load_vector(current_request, current_load_index, load_vector):
    start_time = float(current_request['timestamp'])
    end_time = round(start_time + float(current_request['load']), 3)

    last_period_index = int(math.floor(end_time / period))
    first_period_index = int(math.floor(start_time / period))
    num_of_periods = 1 + last_period_index - first_period_index

    current_load = round(float(current_request['load']), 3)

    if num_of_periods == 1:
        load_vector[first_period_index] = load_vector[first_period_index] + normalize(current_load)

    if num_of_periods == 2:
        first_period_load = (first_period_index + 1) * period - start_time
        second_period_load = current_load - first_period_load

        first_period_load = normalize(first_period_load)
        second_period_load = normalize(second_period_load)

        load_vector[first_period_index] += first_period_load

        if len(load_vector) > first_period_index + 1:
            load_vector[first_period_index + 1] += second_period_load
        else:
            load_vector.append(second_period_load)

    if num_of_periods > 2:
        first_period_load = (first_period_index + 1) * period - start_time
        num_of_full_periods = num_of_periods - 2

        first_period_load = normalize(first_period_load)

        load_vector[current_load_index] = load_vector[current_load_index] + first_period_load

        for index in range(num_of_full_periods):
            current_index = first_period_index + index
            if len(load_vector) > current_index + 1:
                load_vector[current_index] = load_vector[current_index] + 1.0
            else:
                load_vector.append(1.0)

        last_period_load = normalize(current_load - first_period_load - num_of_full_periods)

        if len(load_vector) > last_period_index:
            load_vector[last_period_index] = load_vector[last_period_index] + last_period_load
        else:
            load_vector.append(last_period_load)

    return [round(load, 3) for load in load_vector]


def normalize(load):
    return round(load / float(period), 3)


if __name__ == "__main__":
    generator(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))
