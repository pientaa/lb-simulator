from math import ceil
from math import isnan
import pandas as pd
import numpy as np

PERIOD = 5.0

def estimate_delays(parallel_requests=5):
    print("Delays_estimator started with following params:")
    print("{ parallel_requests: " + str(parallel_requests) + "  } \n")

    load_vectors_df = pd.read_csv("./generator/load_vectors.csv", header=None)

    num_of_samples = load_vectors_df.shape[1]
    shards_on_nodes = pd.read_csv("./simulator/shard_allocated.csv")
    all_requests = pd.read_csv("./generator/requests.csv")
    all_requests = all_requests.rename(columns={"Unnamed: 0": "id"}, inplace=False)

    E_S = all_requests['load'].mean() ## Mean value of load in all requests

    total_delay = 0.0 # Delay for entire cloud
    c_a_estimated = 0.0
    c_a_calculated = 0.0

    for (node, shards) in shards_on_nodes.groupby('node'):
        shards_list = shards['shard'].to_list()
        
        vectors = []
        for index, shard in shards.iterrows():
            vectors.append(load_vectors_df.iloc[shard['shard'] - 1, :].to_list())

        vectors_df = pd.DataFrame(vectors)
        ro_i = pd.DataFrame(vectors_df).values.sum() / (parallel_requests * num_of_samples)

        requests_on_node = all_requests[all_requests['shard'].isin(shards_list)]
        requests_on_node = requests_on_node.assign(period=requests_on_node['timestamp'].map(lambda x: ceil(x / PERIOD)))

        WS_ij = 0 
        T_sum = 0 

        num_of_requests_per_period = [0] * num_of_samples
        for (period, requests) in requests_on_node.groupby('period'):
            num_of_requests_per_period[period - 1] = requests['period'].count()

        c_a_i = requests_on_node['timestamp'].diff().std() / requests_on_node['timestamp'].diff().mean()
        # c_a_i = pd.DataFrame(num_of_requests_per_period).std() / pd.DataFrame(num_of_requests_per_period).mean()
        for (period, requests) in requests_on_node.groupby('period'):
            E_ij_S = requests['load'].mean()
            if(E_ij_S == 0.0):
                continue
            timestamps_list = requests['timestamp'].tolist()
            appear_differences = requests['timestamp'].diff().tolist()
            appear_differences[0] = timestamps_list[0] - PERIOD * (period-1)
            appear_differences.append(PERIOD*(period) - timestamps_list[len(timestamps_list) - 1] )
            c_a_ij = pd.DataFrame(appear_differences)[0].std() / pd.DataFrame(appear_differences)[0].mean()
            c_s_ij = requests['load'].std() / requests['load'].mean()
            ro_ij = (requests['load'].sum() + WS_ij) / (parallel_requests) 

            if(isnan(c_s_ij)):
                c_s_ij = 0.0

            ro_l_ij = PERIOD / ((c_a_ij**2 + c_s_ij**2) * E_ij_S + PERIOD)


            c_a_estimated += ((ro_ij / ro_i) * c_a_i)
            c_a_calculated += c_a_ij

            if(float(ro_ij) < float(ro_l_ij)):
                T = (ro_ij / (1 - ro_ij)) * ((c_a_ij**2 + c_s_ij**2) / 2) * E_ij_S
            else:
                T = 0.5 * PERIOD

            if(float(ro_ij) <= float(ro_l_ij)):
                WS_ij = 0
            else:
                WS_ij = requests['load'].sum() - ro_l_ij * parallel_requests 

            T_sum = T_sum + T
        
        total_delay = total_delay + float(T_sum)


    # print(pd.DataFrame(c_a_calculated))
    # print(pd.DataFrame(c_a_estimated))
    total_delay = total_delay / (num_of_samples * node)
    c_a_calculated = c_a_calculated / (num_of_samples * node)
    c_a_estimated = c_a_estimated / (num_of_samples * node)
    print("c_a_calc: ", c_a_calculated)
    print("c_a_esti: ", c_a_estimated)
    return total_delay, (total_delay / E_S) 

if __name__ == "__main__":
    estimate_delays(5)