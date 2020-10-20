import numpy as np
import sys
import csv
import random
import matplotlib.pyplot as plt
import scipy.stats

def generator(shards, deltaT, N, mean, std):
    vectors = []
    tasks_in_deltaT = []

    for z in range(shards):        
        y = np.random.normal(mean, std, N).tolist()
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
        
        


if __name__== "__main__":
    generator(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))