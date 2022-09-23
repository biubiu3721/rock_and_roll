# -*- coding:utf-8 -*-
"""
Project   : rock_and_roll
File Name : tensor.py
Author    : Focus
Date      : 9/23/2022 12:12 PM
Keywords  : 
Abstract  :
Param     : 
Usage     : py tensor.py
Reference :
"""
import numpy as np
# import matplotlib.pyplot as plt
# import sys

if __name__ == "__main__":
    print("start in main")
    import torch

    # 空， 均匀分布随机数， 0，
    print(torch.empty(5, 3))
    print(torch.rand(5, 3))
    print(torch.zeros(5, 3, dtype=torch.long))

    # list numpy 转 torch
    print(torch.tensor([5, 5, 3]), torch.tensor([5, 5, 3]).dtype, torch.tensor([5, 5, 3]).size()) # default is int64
    print(torch.tensor(np.array([5, 5, 3])))

    # 基本操作
    x = torch.rand(5, 3)
    y = torch.rand(5, 3)
    z = torch.add(x, y)
    print(z)
    print(x[:, 1])

    # reshape
    x = torch.randn(4, 4)
    y = x.view(16)
    z = x.view(-1, 8)
    print(x.size(), y.size(), z.size())

    # numpy 显示互转
    a = torch.ones(5)
    b = a.numpy()
    print(a, b, type(a), type(b))
    a = np.ones(5) # default is float32
    b = torch.from_numpy(a) # default is float64
    print(a, b, type(a), type(b))

