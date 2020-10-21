import numpy as np
import sys
import random

def generator(num_of_shards, period, size_of_vectors, mean, std):
    vectors = []
    tasks_in_period = []
    time_marks_in_period = []
    requests = []

    vectors = create_vectors(int(num_of_shards), int(size_of_vectors), float(mean), float(std))

    save_vectors(vectors)
            
    tasks_in_period = count_positive_tasks(int(size_of_vectors), vectors)

    time_marks_in_period = generate_time_stamps(int(size_of_vectors), float(period), tasks_in_period)
    
    requests = create_requests(int(size_of_vectors), tasks_in_period, time_marks_in_period, vectors)
    
    add_indexes_to_requests(requests)

    save_requests(requests)

def create_vectors(num_of_shards, size_of_vectors, mean, std):
    vectors = []
    for shard in range(num_of_shards): 
        loads = []    
        loads = np.random.normal(mean, std, size_of_vectors).tolist()
        loads = [round(load, 3) for load in loads]
        vectors.append(loads)
    return vectors

def save_vectors(vectors):
    vectors_file = open("./generator/vectors.csv","w")
    for vector in vectors:
       vector_string = str(vectors.index(vector)+1) +","+ ','.join(map(str, vector)) + "\n"
       vectors_file.write(vector_string)
    vectors_file.close()

def count_positive_tasks(size_of_vectors, vectors):
    positive_tasks = []
    for column in range(size_of_vectors):
        num_of_positive_loads = 0
        for vector in vectors:
            if(vector[column]>0):
                num_of_positive_loads = num_of_positive_loads + 1
        positive_tasks.append(num_of_positive_loads)
    return positive_tasks

def generate_time_stamps(size_of_vectors, period, tasks_in_period):
    time_marks_in_period = []
    for size_of_vector in range(size_of_vectors):
        period_start = size_of_vector * period
        period_end = (size_of_vector + 1) * period
        time_marks = np.random.uniform(period_start, period_end, tasks_in_period[size_of_vector])
        time_marks = [round(time_mark, 2) for time_mark in time_marks]
        time_marks_in_period.append(time_marks)
    return time_marks_in_period

def create_requests(size_of_vectors, tasks_in_period, time_marks_in_period, vectors):
    requests = []
    for size_of_vector in range(size_of_vectors):
        for vector in vectors:
            if(vector[size_of_vector]>0):
                request = []
                request.append(time_marks_in_period[size_of_vector][vectors.index(vector)])
                request.append(vectors.index(vector)+1)
                request.append(vector[size_of_vector])
                requests.append(request)
                request = []
    return sorted(requests, key=lambda x: x[0])

def add_indexes_to_requests(requests_sorted):
    for request in requests_sorted:
        request.insert(0, requests_sorted.index(request) + 1)

def save_requests(requests):
    requests_file = open("./generator/requests.csv","w")
    requests_file.write("id, timestamp, shard, size \n")
    for request in requests:
        request_string = ','.join(map(str, request)) + "\n"
        requests_file.write(request_string)
    requests_file.close()













if __name__== "__main__":
    generator(int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))