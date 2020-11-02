import numpy as np
import pandas as pd

def simulator():
    requests = pd.read_csv("./generator/requests.csv")

    print(requests)

if __name__ == "__main__":
    simulator()