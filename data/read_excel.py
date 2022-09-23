import sys
import pandas as pd
import numpy as np

def read_excel(name):
    pd_data = pd.read_excel(name)
    data = pd_data.to_numpy()[1:]
    shape = data.shape
    data = np.array(list(range(shape[0])) * 8).reshape(shape) + np.random.randn(shape[0], shape[1])
    print(data)
    return data

if __name__ == "__main__":
    read_excel("./data_493.xlsx")