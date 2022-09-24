from data.read_excel import *
import numpy as np
import torch
import torch.nn as nn
from model import *
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.utils.data as Data

class DataSet(Data.Dataset):
    def __init__(self, values, targets):

        self.data = values
        self.targets = targets.reshape(-1, 1)
        self.data = self.data.astype(np.float32)
        self.targets = self.targets.astype(np.float32)

    def __getitem__(self, index):
        data = self.data[index]
        targets = self.targets[index]
        return data, targets

    def __len__(self):
        return len(self.data)

class TrainValidSet:
    def __init__(self, train_set, valid_set):
        self.train_set = train_set
        self.valid_set = valid_set

class RockData:
    def __init__(self, _sample):
        self.total_values = _sample[:, 0:-1] # line 0~6 is value
        self.total_labels = _sample[:, -1]  # line 7 is label
        self.total_num = _sample.shape[0]
        self.dim = _sample.shape[1] - 1

    def get_random_set(self, train_num):
        ids = [i for i in range(self.total_num)]
        np.random.shuffle(ids)
        train_idx = ids[0:train_num]
        valid_idx = ids[train_num:]
        train_set = DataSet(self.total_values[train_idx],
                            self.total_labels[train_idx])
        valid_set = DataSet(self.total_values[valid_idx],
                            self.total_labels[valid_idx])

        return TrainValidSet(train_set, valid_set)

    def fold_samples(self, fold_num = 5):
        self.folders = []
        ids = np.array(range(self.total_num))
        np.random.shuffle(ids)
        each_num = int(self.total_num / fold_num)
        for i in range(fold_num):
            start = i * each_num
            end = min((i + 1) * each_num, self.total_num)
            valid_ids = ids[start: end]
            mask = np.ones(self.total_num, dtype=bool)
            mask[start: end] = False
            train_ids = ids[mask]
            train_set = DataSet(self.total_values[train_ids],
                                self.total_labels[train_ids])
            valid_set = DataSet(self.total_values[valid_ids],
                                self.total_labels[valid_ids])
            self.folders.append(TrainValidSet(train_set, valid_set))


def min_max_scaler(data):
    data = data.reshape(data.shape[0], -1)
    for i in range(data.shape[1]):

        print("min_max_scaler")
        amin = np.min(data[:, i])
        amax = np.max(data[:, i])
        print(" min, max", amin, amax)
        data[:, i] = (data[:, i] - amin)/(amax-amin)
    return data

def train(train_valid_data):
    PATH = './model.pth'
    batch_size = 32
    shuffle = False
    value = torch.from_numpy(min_max_scaler(train_valid_data.train_set.data))
    targets = torch.from_numpy(min_max_scaler(train_valid_data.train_set.targets))
    torch_dataset = Data.TensorDataset(value, targets)
    tor_train_set = DataLoader(torch_dataset, batch_size=batch_size)
    input_dim = 7
    output_dim = 1
    net = LinearRegressionModel(input_dim, output_dim)
    criterion = nn.MSELoss()

    optimizer = torch.optim.SGD(net.parameters(), lr=net.hyper["lr"],  momentum=net.hyper["momentum"])
    loss_set = []
    for epoch in range(1000):
        running_loss = 0.0
        for i, [inputs, labels] in enumerate(tor_train_set):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_set.append(loss.item())
        if epoch % 200 == 0:
            print(loss.item())
            print(f'[{epoch}] loss: {loss.item():.5f}')
    print('Finished Training')

    # test
    x = min_max_scaler(train_valid_data.valid_set.data)
    y = min_max_scaler(train_valid_data.valid_set.targets).reshape(-1)
    y_ = net(torch.from_numpy(x)).data.numpy().reshape(-1)
    idx = np.argsort(y)
    y = y[idx]
    y_ = y_[idx]
    print(y, y_)
    plt.plot(y)
    plt.plot(y_)
    torch.save(net.state_dict(), PATH)
    # print(loss_set)
    # plt.plot(loss_set)
    plt.show()


if __name__ == "__main__":
    data_file = "./data/data_493.xlsx"
    sample = read_excel(data_file)

    rock_data_inst = RockData(sample)
    rock_data_inst.get_random_set(int(493 * 0.9))
    rock_data_inst.fold_samples(5)
    train(rock_data_inst.folders[0])
