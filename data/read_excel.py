import sys
import pandas as pd

def read_excel(name):
    pd_data = pd.read_excel(name)
    data = pd_data.to_numpy()[1:]
    return data

if __name__ == "__main__":
    read_excel("./data_493.xlsx")