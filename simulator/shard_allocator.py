import numpy as np
import pandas as pd
import sys
import numpy as np

def shard_allocator():

    if (not algorithm in ["random", "sequential", "SALP"]):
        sys.exit("Pass one of allocation algorithms: random/sequential/SALP as third param.")

    if (num_of_shards  < 100 * num_of_nodes):
        sys.exit("There should be at least 100 times more shards than nodes.")

    if (algorithm == "random"):
        random_allocation()

    if (algorithm == "sequential"):
        sequential_allocation()    

    if (algorithm == "SALP"):
        SALP_allocation()    

def random_allocation():
    shards_on_nodes = []
    current_node = 1

    shards_shuffled = np.random.choice(range(1, num_of_shards + 1), num_of_shards, replace=False)

    for i in range(len(shards_shuffled)):
        shard = shards_shuffled[i]

        shards_on_nodes.append([shard, current_node])

        if ((i + 1) % 100 == 0):
            current_node += 1

    shards_on_nodes_df = pd.DataFrame(shards_on_nodes, columns=["shard", "node"])
    shards_on_nodes_df.sort_values('shard').to_csv("./simulator/shard_allocated.csv", index=False)

def sequential_allocation():
    shards_on_nodes = []
    current_node = 1

    for shard in range(num_of_shards):

        shards_on_nodes.append([shard + 1, current_node])

        if ((shard + 1) % 100 == 0):
            current_node += 1

    shards_on_nodes_df = pd.DataFrame(shards_on_nodes, columns=["shard", "node"])
    shards_on_nodes_df.to_csv("./simulator/shard_allocated.csv", index=False)

def SALP_allocation():
    # TODO:
    print("NOT IMPLEMENTED YET")

if __name__ == "__main__":
    num_of_shards = int(sys.argv[1])
    num_of_nodes = int(sys.argv[2])
    algorithm = str(sys.argv[3])

    shard_allocator()