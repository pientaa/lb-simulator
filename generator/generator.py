import numpy as np
import sys
import random

def generator(num_of_shards, delta_t, size_of_vectors, mean, std):
    vectors = []
    tasks_in_delta_t = []
    time_marks_in_delta_t = []
    requests = []

    vectors = create_vectors(int(num_of_shards), int(size_of_vectors), float(mean), float(std))

    save_vectors(vectors)
            
    tasks_in_delta_t = count_positive_tasks(int(size_of_vectors), vectors)

    time_marks_in_delta_t = generate_time_stamps(int(size_of_vectors), float(delta_t), tasks_in_delta_t)
    
    requests = create_requests(int(size_of_vectors), tasks_in_delta_t, time_marks_in_delta_t, vectors)

    requests_sorted = sorted(requests, key=lambda x: x[0])
    
    add_indexes_to_requests(requests_sorted)

    save_requests(requests_sorted)

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
    tasks_in_delta_t = []
    for column in range(size_of_vectors):
        num_of_positive_loads = 0
        for vector in vectors:
            if(vector[column]>0):
                num_of_positive_loads = num_of_positive_loads + 1
        tasks_in_delta_t.append(num_of_positive_loads)
    return tasks_in_delta_t

def generate_time_stamps(size_of_vectors, delta_t, tasks_in_delta_t):
    time_marks_in_delta_t = []
    for size_of_vector in range(size_of_vectors):
        delta_t_start = size_of_vector * delta_t
        delta_t_end = (size_of_vector + 1) * delta_t
        time_marks = np.random.uniform(delta_t_start, delta_t_end, tasks_in_delta_t[size_of_vector])
        time_marks = [round(time_mark, 2) for time_mark in time_marks]
        time_marks_in_delta_t.append(time_marks)
    return time_marks_in_delta_t

def create_requests(size_of_vectors, tasks_in_delta_t, time_marks_in_delta_t, vectors):
    requests = []
    for size_of_vector in range(size_of_vectors):
        task_iterator = tasks_in_delta_t[size_of_vector] - 1
        for vector in vectors:
            if(vector[size_of_vector]>0):
                tmp_req = []
                tmp_req.append(time_marks_in_delta_t[size_of_vector][task_iterator])
                tmp_req.append(vectors.index(vector)+1)
                tmp_req.append(vector[size_of_vector])
                task_iterator = task_iterator - 1
                requests.append(tmp_req)
                tmp_req = []
    return requests

def add_indexes_to_requests(requests_sorted):
    for request in requests_sorted:
        request.insert(0, requests_sorted.index(request) + 1)

def save_requests(requests_sorted):
    requests_file = open("./generator/requests.csv","w")
    requests_file.write("id, timestamp, shard, size \n")
    for request in requests_sorted:
        request_string = ','.join(map(str, request)) + "\n"
        requests_file.write(request_string)
    requests_file.close()


if __name__== "__main__":
    generator(int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))