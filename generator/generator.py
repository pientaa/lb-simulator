#import numpy
import sys
import csv
import random
print("Hello")

def generator(shards, deltaT, N, mean, std):
    print("Hello")
    tmp_List = []
    vectors = []

    for x in range(1, shards+1):
        tmp_List.append(x)
        for y in range(1, N):
            tmp_List.append(random.randint(0, 10))
        vectors.append(tmp_List)
        tmp_List = []


    vectors_file = open("vectors.csv","w")
    for x in vectors:
        vector_string = ','.join(map(str, x)) + "\n"
        vectors_file.write(vector_string)





    

    

if __name__== "__main__":
    generator(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))