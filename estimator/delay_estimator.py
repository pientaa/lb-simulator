from math import ceil
import pandas as pd

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

    total_delay = 0 # Delay for entire cloud

    for (node, shards) in shards_on_nodes.groupby('node'):
        shards_list = shards['shard'].to_list()
        
        ## Calculating load vectors for node and ro_i
        vectors = []
        for index, shard in shards.iterrows():
            vectors.append(load_vectors_df.iloc[shard['shard'] - 1, :].to_list())

        vectors_df = pd.DataFrame(vectors)
        ro_i = pd.DataFrame(vectors_df).values.sum() / (parallel_requests * num_of_samples)

        requests_on_node = all_requests[all_requests['shard'].isin(shards_list)]
        requests_on_node = requests_on_node.assign(period=requests_on_node['timestamp'].map(lambda x: ceil(x / PERIOD)))

        WS_i = 0 ## value of accumulated job for next interval
        T_sum = 0 ## Delay in node

        ## Below lines and loop are calculating num of requests in each interval.
        ## After that, the c_a_i factor is calculating (factor for node)
        ## We need these, if we are calculating delays as a mean value, not an actual
        num_of_requests_per_period = [0] * num_of_samples
        for (period, requests) in requests_on_node.groupby('period'):
            num_of_requests_per_period[period - 1] = requests['period'].count()
        c_a_i = pd.DataFrame(num_of_requests_per_period).std() / pd.DataFrame(num_of_requests_per_period).mean()
        c_s_i = 0.5 ## this factor is set "from the mountain"

        for (period, requests) in requests_on_node.groupby('period'):
            E_ij_S = requests['load'].mean() ## Mean load of requests in time interval
            ro_ij = ((requests['load'].sum() + WS_i)/ PERIOD) / (num_of_samples * parallel_requests) ## ro_ij is the value of ro_i but in "j" interval; not sure about WS_i..
            
            ## Below line is calculating c_a_ij factor as a mean value of c_a_i factor for node
            ## c_a_ij factor is impossible to calculate manually, because we have information about only one interval
            c_a_ij = (ro_ij / ro_i ) * c_a_i

            ## Of course we can calculate manually c_s_ij value for the i-node and j-interval, but what for c_a_ij?
            c_s_ij = requests['load'].std() / requests['load'].mean()
            # c_s_ij = (ro_ij / ro_i) * c_s_i
            

            ## Next two lines are calculating the same, but in first we are using mean value of request in interval
            # In the second line we are using mean value of all requests
            # ro_l_ij = PERIOD / ((c_a_ij**2 + c_s_ij**2) * E_ij_S + PERIOD)
            ro_l_ij = PERIOD / ((c_a_ij ** 2 + c_s_ij ** 2) * E_S + PERIOD)
            # print("RO_L_IJ: ", float(ro_l_ij))
            # print("RO_IJ: ", ro_ij)
            if(float(ro_ij) < float(ro_l_ij)):
                ## Next two lines are calculating the same, but in first we are using mean value of request in interval
                ## In the second line we are using mean value of all requests
                # T = (ro_ij / (1 - ro_ij)) * ((c_a_ij**2 + c_s_ij**2) / 2) * E_ij_S
                T = (ro_ij / (1 - ro_ij)) * ((c_a_ij**2 + c_s_ij**2) / 2) * E_S
            else:
                T = 0.5 * PERIOD

            if(float(ro_ij) <= float(ro_l_ij)):
                WS_i = 0
            else:
                # print("flaga")
                WS_i = requests['load'].sum() - ro_l_ij  ## Don't understand that. Am I calculating job in next period correct? I used dr's equation...
                                                        ## Maybe we need multiply ro_l_ij by PERIOD?

            T_sum = T_sum + T
        
        total_delay = total_delay + float(T_sum)

    # print("Total delay: ", total_delay)
    print((total_delay / (num_of_samples * PERIOD)) * 100.0)
    # print(total_delay / E_S)
    # return total_delay, (total_delay / E_S)  ## Dr's equation, this has no sense!
    return total_delay, (total_delay / (num_of_samples * PERIOD)) * 100.0 ## IMO our way is better!

if __name__ == "__main__":
    estimate_delays(5)