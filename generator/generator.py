#import numpy
import sys
import csv
import random

def generator(shards, deltaT, N, mean, std):
    tmp_list = []
    vectors = []

    for x in range(shards):
        for y in range(N):
            tmp_list.append(random.randint(0, 10))
        vectors.append(tmp_list)
        tmp_list = []


    vectors_file = open("./generator/vectors.csv","w")
    for x in vectors:
        vector_string = str(vectors.index(x)+1) +","+ ','.join(map(str, x)) + "\n"
        vectors_file.write(vector_string)





    

    

if __name__== "__main__":
    generator(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))