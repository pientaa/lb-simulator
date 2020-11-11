import numpy as np
import pandas as pd
import sys
from generator.generator import generator 
from simulator.shard_allocator import shard_allocator
from simulator.simulator import simulator

def experiment_executor():
    experiment = int(input("Which experiment?:"))
    while(not experiment in [1,2,3]):
        experiment = int(input("Which experiment? (Enter number from 1 to 3):"))

    num_of_shards = int(input("Num of shards:"))
    num_of_samples = 100
    period = 5.0
    parallel_requests = 5
    num_of_nodes = round(num_of_shards / 10)

    # for i in range(num_of_nodes, num_of_shards, num_of_nodes):

    generator(num_of_shards, num_of_samples, period, num_of_shards/2, 1.0)
    print("Load vectors and requests generated...")
    shard_allocator(num_of_shards, num_of_nodes, "random")
    print("Shards allocated...")
    simulator(parallel_requests, period)
    print("Simulator ended...")

if __name__ == "__main__":

    experiment_executor()