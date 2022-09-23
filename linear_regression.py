# -*- coding:utf-8 -*-
"""
Project   : rock_and_roll
File Name : linear_regression.py
Author    : Focus
Date      : 9/23/2022 12:45 PM
Keywords  : 
Abstract  :
Param     : 
Usage     : py linear_regression.py
Reference :
"""
import numpy as np
import torch
import torch.nn as nn
from data.read_excel import *
import matplotlib.pyplot as plt
# import sys

# 0.88
# class LinearRegressionModel(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(LinearRegressionModel, self).__init__()
#         self.h_dim1 = 32
#         self.h_dim2 = 128
#         self.h_dim3 = 256
#         self.h_dim4 = 128
#         self.fc1 = nn.Linear(input_dim, self.h_dim1)
#         self.fc2 = nn.Linear(self.h_dim1, self.h_dim2)
#         self.fc3 = nn.Linear(self.h_dim2, self.h_dim3)
#         self.fc4 = nn.Linear(self.h_dim3, self.h_dim4)
#         self.fc5 = nn.Linear(self.h_dim4, output_dim)
#         self.act1 = torch.relu
#         self.act2 = torch.relu
#         self.act3 = torch.relu
#         self.act4 = torch.sigmoid
#
#     def forward(self, x):
#         x =  self.fc1(x)
#         x = self.act1(x)
#         x = self.fc2(x)
#         x = self.act2(x)
#         x = self.fc3(x)
#         x = self.act3(x)
#         x = self.fc4(x)
#         x = self.act4(x)
#         x = self.fc5(x)
#         return x
# 0.86
# class LinearRegressionModel(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(LinearRegressionModel, self).__init__()
#         self.h_dim1 = 32
#         self.h_dim2 = 128
#         self.h_dim3 = 256
#         self.h_dim4 = 256
#         self.h_dim5 = 64
#         self.fc1 = nn.Linear(input_dim, self.h_dim1)
#         self.fc2 = nn.Linear(self.h_dim1, self.h_dim2)
#         self.fc3 = nn.Linear(self.h_dim2, self.h_dim3)
#         self.fc4 = nn.Linear(self.h_dim3, self.h_dim4)
#         self.fc5 = nn.Linear(self.h_dim4, self.h_dim5)
#         self.fc6 = nn.Linear(self.h_dim5, output_dim)
#         self.act1 = torch.relu
#         self.act2 = torch.relu
#         self.act3 = torch.relu
#         self.act4 = torch.relu
#         self.act5 = torch.sigmoid
#
#     def forward(self, x):
#         x =  self.fc1(x)
#         x = self.act1(x)
#         x = self.fc2(x)
#         x = self.act2(x)
#         x = self.fc3(x)
#         x = self.act3(x)
#         x = self.fc4(x)
#         x = self.act4(x)
#         x = self.fc5(x)
#         x = self.act5(x)
#         x = self.fc6(x)
#         return x
# resnet 0.00015
# class LinearRegressionModel(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(LinearRegressionModel, self).__init__()
#         self.hyper = dict()
#         self.hyper["lr"] = 0.002
#         self.hyper["epochs"] = 20000
#         self.hyper["momentum"] = 0.9
#         self.h_dim1 = 64
#         self.h_dim2 = 64
#         self.h_dim3 = 64
#         self.h_dim4 = 64
#         self.h_dim5 = 64
#         self.h_dim6 = 64
#         self.fc1 = nn.Linear(input_dim, self.h_dim1)
#         self.fc2 = nn.Linear(self.h_dim1, self.h_dim2)
#         self.fc3 = nn.Linear(self.h_dim2, self.h_dim3)
#         self.fc4 = nn.Linear(self.h_dim3, self.h_dim4)
#         self.fc5 = nn.Linear(self.h_dim4, self.h_dim5)
#         self.fc6 = nn.Linear(self.h_dim5, self.h_dim6)
#         self.fc7 = nn.Linear(self.h_dim6, output_dim)
#         self.act1 = torch.relu
#         self.act2 = torch.relu
#         self.act3 = torch.relu
#         self.act4 = torch.relu
#         self.act5 = torch.relu
#         self.act6 = torch.sigmoid
#         self.bn1 = nn.BatchNorm1d(self.h_dim1, eps= 1e-05, momentum= 0.01, affine= True, track_running_stats= True)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x0 = self.bn1(self.act1(x))
#         x = self.fc2(x0)
#         x = x0 + self.act2(x)
#         x0 = self.fc3(x)
#         x = self.act3(x0)
#         x = x0 + self.fc4(x)
#         x0 = self.act4(x)
#         x = self.fc5(x0)
#         x = x0 + self.act5(x)
#         x0 = self.fc6(x)
#         x = self.act6(x0)
#         x = self.fc7(x)
#         return x

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.hyper = dict()
        self.hyper["lr"] = 0.002
        self.hyper["epochs"] = 20000
        self.hyper["momentum"] = 0.9
        self.h_dim1 = 64
        self.h_dim2 = 64
        self.h_dim3 = 64
        self.h_dim4 = 64
        self.h_dim5 = 64
        self.h_dim6 = 64
        self.fc1 = nn.Linear(input_dim, self.h_dim1)
        self.fc2 = nn.Linear(self.h_dim1, self.h_dim2)
        self.fc3 = nn.Linear(self.h_dim2, self.h_dim3)
        self.fc4 = nn.Linear(self.h_dim3, self.h_dim4)
        self.fc5 = nn.Linear(self.h_dim4, self.h_dim5)
        self.fc6 = nn.Linear(self.h_dim5, self.h_dim6)
        self.fc7 = nn.Linear(self.h_dim6, output_dim)
        self.act1 = torch.relu
        self.act2 = torch.relu
        self.act3 = torch.relu
        self.act4 = torch.relu
        self.act5 = torch.relu
        self.act6 = torch.sigmoid
        self.bn1 = nn.BatchNorm1d(self.h_dim1, eps= 1e-05, momentum= 0.01, affine= True, track_running_stats= True)

    def forward(self, x):
        x = self.fc1(x)
        x0 = self.bn1(self.act1(x))
        x = self.fc2(x0)
        x = x0 + self.act2(x)
        x0 = self.fc3(x)
        x = self.act3(x0)
        x = x0 + self.fc4(x)
        x0 = self.act4(x)
        x = self.fc5(x0)
        x = x0 + self.act5(x)
        x0 = self.fc6(x)
        x = self.act6(x0)
        x = self.fc7(x)
        return x

def min_max_scaler(data):
    print("min_max_scaler")
    amin = np.min(data)
    amax = np.max(data)
    print(" min, max", amin, amax)
    data = (data - amin)/(amax-amin)
    return data

def case1():
    """
    线性回归模型

    """
    start_dim, end_dim = 0, 7
    input_dim = end_dim - start_dim
    num = 493
    output_dim = 1
    """
    numpy array inputs
    """
    # x_values = [i for i in range(11)] * 2
    # x_train = np.array(x_values, dtype=np.float32).reshape(-1, 1)
    #
    # y_values = [2 * i + 1 for i in x_values]
    # y_train = np.array(y_values, dtype=np.float32).reshape(-1, 1)

    """ 
    my data
    """
    data = read_excel("./data/data_493.xlsx")
    x_values = data[0:num, start_dim:end_dim]
    y_values = data[0:num, 7].reshape(-1, 1)
    for i in range(x_values.shape[1]):
        x_values[:,i] = min_max_scaler(x_values[:,i])
    y_values = min_max_scaler(y_values)
    x_train = np.array(x_values, dtype=np.float32).reshape(-1, input_dim)
    y_train = np.array(y_values, dtype=np.float32).reshape(-1, output_dim)

    """
    linear regression model test
    :return:
    """
    # build model

    model = LinearRegressionModel(input_dim, output_dim)
    print(model)

    # params
    hyper = model.hyper
    epochs = hyper['epochs']
    optimizer = torch.optim.SGD(model.parameters(), lr=hyper['lr'], momentum=hyper['momentum'])
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        epoch += 1
        # 转成 tensor
        inputs = torch.from_numpy(x_train)
        labels = torch.from_numpy(y_train)
        # 每一次迭代梯度清零
        optimizer.zero_grad()
        # 前向传播
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新权重参数
        optimizer.step()
        if epoch % 200 == 0:
            print("epoch {}, loss {}".format(epoch, loss.item()))
    # test
    predicted = model(torch.from_numpy(x_train)).data.numpy()
    print(predicted.shape, labels.shape)
    print(predicted[0:20], labels[0:20])
    torch.save(model, 'resnet.pth')
    torch.save(model.state_dict(), "model.pkl")
    model.load_state_dict(torch.load("model.pkl"))
if __name__ == "__main__":
    print("start in main")
    case1()

