from data.read_excel import *
import numpy as np


class DataSet:
    def __init__(self, value, label):
        self.value = value
        self.label = value
        self.num = value.shape[0]
        self.dim = value.shape[1]


class RockData:
    def __init__(self, _sample):
        self.total_values = _sample[:, 0:-1] # line 0~6 is value
        self.total_labels = _sample[:, -1]  # line 7 is label
        self.total_num = _sample.shape[0]
        self.dim = _sample.shape[1] - 1

    def get_random_set(self, train_num):
        index = np.random.shuffle(range(self.total_num))
        train_idx = index[0:train_num]
        valid_idx = index[train_num]
        train_set = DataSet(self.total_values[train_idx[:], :],
                            self.total_labels[train_idx[:], :])
        valid_set = DataSet(self.total_values[valid_idx[:], :],
                            self.total_labels[valid_idx[:], :])
        return train_set, valid_set

    def folder


if __name__ == "__main__":
    data_file = "./data/data_493.xlsx"
    sample = read_excel(data_file)
    rock_data_inst = RockData(sample)
    rock_data_inst.get_random_set(int(493*0.8))

