import pandas as pd


def estimate_delays(parallel_requests=5):
    load_vectors_df = pd.read_csv("./../generator/load_vectors.csv", header=None)
    num_of_shards = load_vectors_df.shape[0]
    num_of_samples = load_vectors_df.shape[1]

    shards_on_nodes = pd.read_csv("./../simulator/shard_allocated.csv")

    for (node, shards) in shards_on_nodes.groupby('node'):
        vectors = []
        for index, shard in shards.iterrows():
            vectors.append(load_vectors_df.iloc[shard['shard'] - 1, :].to_list())
        node_load = pd.DataFrame(vectors).values.sum() / (parallel_requests * num_of_samples)
        print(node_load)


#     TODO: Get mean and std for each node (read requests and group by shards on nodes)


if __name__ == "__main__":
    estimate_delays()
