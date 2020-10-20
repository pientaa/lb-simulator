import numpy as np
import sys
import csv
import random
import matplotlib.pyplot as plt
import scipy.stats

def generator(shards, deltaT, N, mean, std):
    vectors = []
    tasks_in_deltaT = []
    requests = []
    time_marks_in_deltaT = []

    for z in range(shards):        
        y = np.random.normal(mean, std, N).tolist()
        y = [round(num, 3) for num in y]
        vectors.append(y)
        y = []

    vectors_file = open("./generator/vectors.csv","w")
    for z in vectors:
       vector_string = str(vectors.index(z)+1) +","+ ','.join(map(str, z)) + "\n"
       vectors_file.write(vector_string)
    vectors_file.close()
            
    for n in range(N):
        tmp = 0
        for v in vectors:
            if(v[n]>0):
                tmp = tmp + 1
        tasks_in_deltaT.append(tmp)


    for n in range(N):
        tmp_deltaT_start = n * deltaT
        tmp_deltaT_end = (n + 1) * deltaT
        time_marks = np.random.uniform(tmp_deltaT_start, tmp_deltaT_end, tasks_in_deltaT[n])
        time_marks = [round(tm, 2) for tm in time_marks]
        time_marks_in_deltaT.append(time_marks)

    tmp_req = []
    for n in range(N):
        task_iterator = tasks_in_deltaT[n] - 1
        for v in vectors:
            if(v[n]>0):
                tmp_req.append(time_marks_in_deltaT[n][task_iterator])
                tmp_req.append(vectors.index(v)+1)
                tmp_req.append(v[n])
                task_iterator = task_iterator - 1
                requests.append(tmp_req)
                tmp_req = []
    
    requests_sorted = sorted(requests, key=lambda x: x[0])
    
    for x in requests_sorted:
        x.insert(0, requests_sorted.index(x) + 1)

    requests_file = open("./generator/requests.csv","w")
    requests_file.write("id, timestamp, shard, size \n")
    for x in requests_sorted:
        request_string = ','.join(map(str, x)) + "\n"
        requests_file.write(request_string)
    requests_file.close()

if __name__== "__main__":
    generator(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))