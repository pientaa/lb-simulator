import numpy as np
import pandas
import sys
import random

def generator(num_of_shards, num_of_samples, period, mean, std):
    vectors = []
    time_marks_in_period = []
    requests = []

    vectors = create_vectors(int(num_of_shards), int(num_of_samples), float(mean), float(std))

    time_marks_in_period = generate_time_stamps(int(num_of_samples), float(period))
    
    requests = create_requests(time_marks_in_period, vectors)
    
    add_indexes_to_requests(requests)

    save_requests(requests)

    generate_load_vectors(requests, period, num_of_shards)

def create_vectors(num_of_shards, num_of_samples, mean, std):
    vectors = []
    for shard in range(num_of_shards): 
        loads = []    
        loads = non_negative_random_normal(mean, std, num_of_samples)
        loads = [round(load, 3) for load in loads]
        vectors.append(loads)
    return vectors

def non_negative_random_normal(mean, std, num_of_samples):
    loads = np.random.normal(mean, std, num_of_samples).tolist()
    return(loads if all(load >=0 for load in loads) else non_negative_random_normal(mean, std, num_of_samples))

def generate_time_stamps(num_of_samples, period):
    time_marks_in_period = []
    for index in range(num_of_samples):
        period_start = index * period
        period_end = (index + 1) * period
        time_marks = np.random.uniform(period_start, period_end, num_of_samples)
        time_marks = [round(time_mark, 2) for time_mark in time_marks]
        time_marks_in_period.append(time_marks)
    return time_marks_in_period

def create_requests(time_marks_in_period, vectors):
    requests = []
    request = []
    for vector in vectors:
        for load in vector:
            # TODO: ELIMINATE .index
            shard_id = vectors.index(vector) + 1
            load_index = vector.index(load)
            timestamp = time_marks_in_period[load_index][shard_id - 1]

            request.append(timestamp)
            request.append(shard_id)
            request.append(load)

            requests.append(request)
            request = []

    return sorted(requests, key=lambda x: x[0])

def add_indexes_to_requests(sorted_requests):
    for request in sorted_requests:
        request.insert(0, sorted_requests.index(request) + 1)

def save_requests(requests):
    requests_file = open("./generator/requests.csv","w")
    requests_file.write("id, timestamp, shard, load \n")
    for request in requests:
        request_string = ','.join(map(str, request)) + "\n"
        requests_file.write(request_string)
    requests_file.close()

def generate_load_vectors(requests, period, num_of_shards):
    
    requests_df = pandas.DataFrame(requests, columns=['id', 'timestamp', 'shard', 'load'])

    for (shard, group) in requests_df.groupby('shard'):
        load_vector = [0] * num_of_shards

        i = 0
        for load in load_vector:
            current_request = group[(group['timestamp'] >= period * i) & (group['timestamp'] < period * (i + 1))]

            start_time = float(current_request['timestamp'])
            end_time = round(start_time + float(current_request['load']), 3)
            
            # TODO: Find all loads that need to be incremented by apppropriate load value
            # TODO: Increment load values

            i = i + 1


if __name__== "__main__":
    period = float(sys.argv[3])
    generator(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))