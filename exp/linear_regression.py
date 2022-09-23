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
# import matplotlib.pyplot as plt
# import sys


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

def case1():
    """
    线性回归模型

    """

    # numpy array inputs
    x_values = [i for i in range(11)] * 2
    x_train = np.array(x_values, dtype=np.float32).reshape(-1, 1)
    print(x_train.shape)

    y_values = [2 * i + 1 for i in x_values]
    y_train = np.array(y_values, dtype=np.float32).reshape(-1, 1)
    print(y_train.shape)

    """
    linear regression model test
    :return:
    """
    # build model
    input_dim = 1
    output_dim = 1
    model = LinearRegressionModel(input_dim, output_dim)
    print(model)

    # params
    epochs = 1000
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
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
        if epoch % 50 == 0:
            print("epoch {}, loss {}".format(epoch, loss.item()))
    # test
    predicted = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()
    print(predicted)
    torch.save(model.state_dict(), "model.pkl")
    model.load_state_dict(torch.load("model.pkl"))
if __name__ == "__main__":
    print("start in main")
    case1()

