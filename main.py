from data.read_excel import *
import numpy as np
import torch
import torch.nn as nn
from model import Net
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

def train(train_valid_data):
    PATH = './model.pth'
    batch_size = 64
    shuffle = True
    lr = 0.00001
    momentum = 0
    value = torch.from_numpy(train_valid_data.train_set.data)
    targets = torch.from_numpy(train_valid_data.train_set.targets)
    torch_dataset = Data.TensorDataset(value, targets)
    tor_train_set = DataLoader(torch_dataset, batch_size=batch_size)
    net = Net()
    for name, parameters in net.named_parameters():  # 打印出每一层的参数的大小
        print(name, ':', parameters, ":", parameters.size())
    criterion = nn.MSELoss()

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_set = []
    for epoch in range(12):
        running_loss = 0.0
        net.train()
        for i, [inputs, labels] in enumerate(tor_train_set):

            print("inputs:", inputs, inputs.shape)
            print("labels:", labels, labels.shape)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            #print("inputs", inputs)
            #print("labels", labels)
            #print("outputs", outputs)
            #print("labels", labels)
            # for parameters in net.parameters():
            #     print(parameters)
            #print("labels", labels)

            print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loss_set.append(loss.item())
            #print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.3f}')
            running_loss = 0
    print('Finished Training')
    for name, parameters in net.named_parameters():  # 打印出每一层的参数的大小
        print(name, ':', parameters, ":", parameters.size())


    torch.save(net.state_dict(), PATH)

    plt.plot(loss_set)
    plt.show()
if __name__ == "__main__":
    data_file = "./data/data_493.xlsx"
    sample = read_excel(data_file)

    rock_data_inst = RockData(sample)
    rock_data_inst.get_random_set(int(493 * 0.8))
    rock_data_inst.fold_samples(5)
    train(rock_data_inst.folders[0])
