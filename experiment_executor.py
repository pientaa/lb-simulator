import os

from generator.generator import generator
from simulator.shard_allocator import shard_allocator
from simulator.simulator import simulator
import pandas as pd
import matplotlib.pyplot as plt

def experiment_executor():
    clear_directory()
    experiment = int(input("Which experiment?:"))
    # experiment = 1
    while experiment not in [1, 2, 3]:
        experiment = int(input("Which experiment? (Enter number from 1 to 3):"))

    # algorithms = "SALP"

    algorithms = str(input("Which allocation algorithm?:"))
    while algorithms not in ["random", "sequential", "SALP", "all"]:
        algorithms = str(input("Which allocation algorithm? (random/sequential/SALP):"))

    if algorithms == "all":
        algorithms = ["random", "sequential", "SALP"]
    else:
        algorithms = [algorithms]

    num_of_shards = int(input("Num of shards:"))
    # num_of_shards = 300
    num_of_samples = 100
    period = 5.0
    parallel_requests = 5
    num_of_nodes = round(num_of_shards / 25)
    max_num_of_nodes = round(num_of_shards / 10) + 1

    clear_directory()
    requests, load_vectors = generator(num_of_shards, num_of_samples, period, 2.0, num_of_shards / 16.0)

    requests.to_csv('./experiments/requests.csv')
    requests.to_csv('./generator/requests.csv')

    for vector in load_vectors:
        save_load_vector(vector)

    # DataFrame przechowujacy wyniki
    delays_df = pd.DataFrame(columns = ['algorithm', 'nodes', 'sum_of_delay', 'delay_percentage'])



    for nodes in range(num_of_nodes, max_num_of_nodes, 1):
        for algorithm in algorithms:
            shard_allocation(num_of_shards, nodes, algorithm)
            #TODO:
            #Funkcja wyznaczająca poziom niezrównoważnia. Przyda się przerobić shard_allocator, który zwróci DataFrame z zalokowanymi shardami.
            #Łatwiej będzie wtedy wyznaczyć poziomy niezrównoważenia, nie trzeba będzie na bazie csv'ek z load_vectorami grupować shardów i zliczać wektory obciązenia
            #przypadające na węzeł

            requests_completed_df = simulation(parallel_requests, period, nodes, algorithm)

            complete_processing_time = num_of_samples * period

            #"odcięcie" requestów, które rozpoczęły się po oknie przetwarzania
            requests_completed_df = requests_completed_df[requests_completed_df['timestamp'] < complete_processing_time]

            #iteracja po requestach, których czas zakończenia > okno czasowe
            for index, row in requests_completed_df[requests_completed_df['actual_end_time'] > complete_processing_time].iterrows():
                requests_completed_df.at[index, 'actual_end_time'] = complete_processing_time #zmiana czasu zakończenia requastu na czas okna przetwarzania
                new_delay = complete_processing_time - requests_completed_df[requests_completed_df.index == index]['expected_end_time'].item() #Przeliczenie delay'a


                #może się zdarzyć, że expected_time > okno przetwarzania
                #wtedy delay = 0
                if(new_delay >= 0):
                    requests_completed_df.at[index, 'delay'] = new_delay
                else:
                    requests_completed_df.at[index, 'delay'] = 0
                
            #Wyznaczenie procentowego udziału delay w oknie przetwarzania
            sum_of_delay = requests_completed_df['delay'].sum()
            delay_percentage = (sum_of_delay / complete_processing_time) * 100.0

            to_append = {'algorithm' : algorithm, 'nodes' : nodes, 'sum_of_delay' : sum_of_delay, 'delay_percentage' : delay_percentage}

            delays_df = delays_df.append(to_append, ignore_index=True)
    generate_delays_plots(delays_df)        

def generate_delays_plots(delays_df):

    #TODO:
    #  Dopracować wykresy (kolory, labele, przeskalować osie itp itd.)
    for group in delays_df['algorithm'].unique():
        plt.plot(delays_df[delays_df['algorithm'] == group]['nodes'].tolist(),
                delays_df[delays_df['algorithm'] == group]['delay_percentage'].tolist(),
                label = group,
                linewidth=2,)
    plt.show()

def simulation(parallel_requests, period, nodes, algorithm):
    requests_completed_df = simulator(parallel_requests, period)
    requests_completed_df.to_csv('./experiments/' + algorithm + '/requests_completed_' + str(nodes) + '.csv',
                                 index=False)
    return requests_completed_df


def shard_allocation(num_of_shards, nodes, algorithm):
    shard_allocated_df = shard_allocator(num_of_shards, nodes, algorithm)

    shards_allocated = []
    for index, row in shard_allocated_df.iterrows():
        node = row['node']
        for shard in row['shards']:
            shards_allocated.append([node, shard])

    shards_allocated_to_csv = pd.DataFrame(shards_allocated, columns = ['node', 'shard'])

    shards_allocated_to_csv.to_csv('./experiments/' + algorithm + '/shard_allocated_' + str(nodes) + '.csv', index=False)
    shards_allocated_to_csv.to_csv('./simulator/shard_allocated.csv', index=False)

    return shard_allocated_df

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
