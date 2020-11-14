import numpy as np
import pandas as pd
import sys
import os
from generator.generator import generator 
from simulator.shard_allocator import shard_allocator
from simulator.simulator import simulator

def experiment_executor():
    clear_directory()
    experiment = int(input("Which experiment?:"))
    while(not experiment in [1,2,3]):
        experiment = int(input("Which experiment? (Enter number from 1 to 3):"))
    
    algorithm = str(input("Which allocation algorithm?:"))
    while(not algorithm in ["random", "sequential", "SALP", "all"]):
        algorithm = str(input("Which allocation algorithm? (random/sequential/SALP):"))

    num_of_shards = int(input("Num of shards:"))
    num_of_samples = 100
    period = 5.0
    parallel_requests = round(num_of_shards/4)
    num_of_nodes = 2
    max_num_of_nodes = round(num_of_shards / 25)

    clear_directory()
    print("Start generating load vectors and requests...")
    requests, load_vectors = generator(num_of_shards, num_of_samples, period, 1.0, num_of_shards/4)

    requests.to_csv('./experiments/requests.csv')
    requests.to_csv('./generator/requests.csv')

    for vector in load_vectors:
        save_load_vector(vector)
    
    for i in range(num_of_nodes, max_num_of_nodes, 1):
        print("Node: " + str(i))
        if(algorithm == "all"):
            for alg in ["random", "sequential", "SALP"]:
                print("Start shard allocation with following algorithm: " + alg +"...")
                shard_allocated_df = shard_allocator(num_of_shards, i, alg)
                shard_allocated_df.to_csv('./experiments/' + alg + '/shard_allocated_' + str(i) + '.csv', index=False)
                shard_allocated_df.to_csv('./simulator/shard_allocated.csv', index=False)

                print("Start simulation...")
                requests_completed_df = simulator(parallel_requests, period)
                requests_completed_df.to_csv('./experiments/'  + alg + '/requests_completed_' + str(i) + '.csv', index=False)

                complete_processing_time = requests_completed_df.sort_values(by=["actual_end_time"]).tail(1)["actual_end_time"].item()

                delay = (requests_completed_df['delay'].sum() / complete_processing_time) * 100.0

                print(requests_completed_df['delay'].sum())
                print(complete_processing_time)
                print(delay)
        else:
            print("Start shard allocation with following algorithm: " + algorithm +"...")
            shard_allocated_df = shard_allocator(num_of_shards, num_of_nodes, algorithm)
            shard_allocated_df.to_csv('./experiments/shard_allocated.csv', index=False)
            shard_allocated_df.to_csv('./simulator/shard_allocated.csv', index=False)
            
            print("Start simulation...")
            requests_completed_df = simulator(parallel_requests, period)
            requests_completed_df.to_csv('./experiments/requests_completed.csv', index=False)

            complete_processing_time = requests_completed_df.sort_values(by=["actual_end_time"]).tail(1)["actual_end_time"].item()

            delay = (requests_completed_df['delay'].sum() / complete_processing_time) * 100.0

            print(requests_completed_df['delay'].sum())
            print(complete_processing_time)
            print(delay)

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