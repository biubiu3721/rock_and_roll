# -*- coding:utf-8 -*-
"""
Project   : rock_and_roll
File Name : lenet
Author    : Focus
Date      : 9/22/2022 8:21 PM
Keywords  : 
Abstract  :
Param     : 
Usage     : py lenet
Reference :
"""
# import numpy as np
# import matplotlib.pyplot as plt
# import sys

import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch import nn


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    print("start in main")

    train_data = torchvision.datasets.CIFAR10(root="CIFAR10", train=True, transform=torchvision.transforms.ToTensor(),
                                              download=True)
    test_data = torchvision.datasets.CIFAR10(root="CIFAR10", train=False, transform=torchvision.transforms.ToTensor(),
                                             download=True)
    train_data_x = train_data.data
    train_data_y = train_data.targets
    train_num = len(train_data.targets)
    dim = 32 * 32 * 3
    print(type(train_data))
    data = np.array(list(range(train_num)) * (dim + 1)).reshape(train_num, -1) + np.random.randn(train_num, dim + 1)
    train_data_x = torch.tensor(torch.from_numpy(data[:, 0: dim]), dtype=torch.float32)
    train_data_y = [int(i / 100) for i in data[:, dim].tolist()]
    train_data = torch.utils.data.TensorDataset(train_data_x, train_data_y)
    test_data = train_data
    print(train_data_x.shape, train_data_y.shape)

    train_dataloader = DataLoader(train_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)
   # print(train_data.targets)

   # print(train_data.data.shape, len(train_data.classes))
    # 创建网络模型
    module = Module()
    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 优化器
    learning_rate = 1e-2
    optimizer = torch.optim.SGD(module.parameters(), lr=learning_rate)
    # 训练的轮数
    epoch = 12
    # 储存路径
    work_dir = './LeNet'
    # 添加tensorboard
    writer = SummaryWriter("{}/logs".format(work_dir))

    for i in range(epoch):
        print("-------epoch  {} -------".format(i + 1))
        # 训练步骤
        module.train()
        for step, [imgs, targets] in enumerate(train_dataloader):
            outputs = module(imgs)
            loss = loss_fn(outputs, targets)

            # 优化器
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_step = len(train_dataloader) * i + step + 1
            if train_step % 100 == 0:
                print("train time：{}, Loss: {}".format(train_step, loss.item()))
                writer.add_scalar("train_loss", loss.item(), train_step)

        # 测试步骤
        module.eval()
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for imgs, targets in test_dataloader:
                outputs = module(imgs)
                loss = loss_fn(outputs, targets)
                total_test_loss = total_test_loss + loss.item()
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy = total_accuracy + accuracy

        print("test set Loss: {}".format(total_test_loss))
        print("test set accuracy: {}".format(total_accuracy / len(test_data)))
        writer.add_scalar("test_loss", total_test_loss, i)
        writer.add_scalar("test_accuracy", total_accuracy / len(test_data), i)

        torch.save(module, "{}/module_{}.pth".format(work_dir, i + 1))
        print("saved epoch {}".format(i + 1))

    writer.close()