from math import ceil

import pandas as pd

PERIOD = 5.0


def estimate_delays(parallel_requests=5):
    print("Delays_estimator started with following params:")
    print("{ parallel_requests: " + str(parallel_requests) + "  } \n")

    load_vectors_df = pd.read_csv("./generator/load_vectors.csv", header=None)

    num_of_samples = load_vectors_df.shape[1]

    shards_on_nodes = pd.read_csv("./simulator/shard_allocated.csv")
    requests = pd.read_csv("./generator/requests.csv")
    requests = requests.rename(columns={"Unnamed: 0": "id"}, inplace=False)

    T_sum = 0

    for (node, shards) in shards_on_nodes.groupby('node'):
        vectors = []
        for index, shard in shards.iterrows():
            vectors.append(load_vectors_df.iloc[shard['shard'] - 1, :].to_list())

        requests_on_node = requests[requests['shard'].isin(shards['shard'].to_list())]
        requests_on_node = requests_on_node.assign(period=requests_on_node['timestamp'].map(lambda x: ceil(x / PERIOD)))
        requests_per_period = [0] * num_of_samples

        for (period, requests) in requests_on_node.groupby('period'):
            requests_per_period[period - 1] = requests['period'].count()

        node_load = pd.DataFrame(vectors).values.sum() / (parallel_requests * num_of_samples)

        requests_per_period_std = pd.DataFrame(requests_per_period).values.std()

        if requests_per_period_std <= 0.0:
            c_a = 0.0
        else:
            c_a = pd.DataFrame(requests_per_period).std() / pd.DataFrame(requests_per_period).mean()

        T = pd.DataFrame(vectors).values.sum() * abs(node_load / (1 - node_load)) * (c_a * c_a / 2.0)

        # print("node_load:")
        # print(node_load)
        # print("c_a:")
        # print(c_a)
        # print("T:")
        # print(T)

        T_sum = T_sum + T

    return T_sum, (T_sum / (num_of_samples * PERIOD)) * 100.0
