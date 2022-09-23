# -*- coding:utf-8 -*-
"""
Project   : rock_and_roll
File Name : autograd
Author    : Focus
Date      : 9/23/2022 12:26 PM
Keywords  : 
Abstract  :
Param     : 
Usage     : py autograd
Reference :
"""
import numpy as np
# import matplotlib.pyplot as plt
# import sys
import torch


def case1():
    x = torch.randn(3, 4, requires_grad=True)
    print(x)
    x = torch.randn(3, 4)
    x.requires_grad = True
    b = torch.randn(3, 4, requires_grad=True)
    t = x + b
    y = t.sum()
    print("t", t)
    print("y", y)
    y.backward()
    print(x.requires_grad, b.requires_grad, t.requires_grad)
    print(b.grad)



def case2():
    # refer to "./autograd.png"
    x = torch.rand(1)
    b = torch.rand(1, requires_grad=True)
    w = torch.rand(1, requires_grad=True)
    y = w * x
    z = y + b
    print(x.requires_grad, b.requires_grad, w.requires_grad, y.requires_grad, z.is_leaf)
    print(x.is_leaf, b.is_leaf, w.is_leaf, y.is_leaf, z.is_leaf)
    # 为什么梯度默认是累加操作呢？
    z.backward(retain_graph=True)
    print(b.grad, w.grad, y.grad)

if __name__ == "__main__":
    print("start in main")
    case1()
    case2()