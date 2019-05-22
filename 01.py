import torch 
import torchvision 
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

# 1. autograd例子
# create tensors 创建张量

# requires_grad:设置为True是允许精细的排除子图 提高计算效率
x = torch.tensor(1.,requires_grad=True)
w = torch.tensor(2.,requires_grad=True)
b = torch.tensor(3.,requires_grad=True)

# 建立一个计算图
y = w*x + b 

# 计算梯度 
# 疑问：梯度是如何计算的？？？？
y.backward()
print(x.grad)
print(w.grad)
print(b.grad)

# 2. autograd例子

# 创建二维的tensor
# randn 表示从标准正态分布中抽取随机数
x = torch.randn(10,3)
y = torch.randn(10,2)

# 建立一个全连接层
linear = nn.Linear(3,2)