# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:23:07 2020

@author: Vrushali
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

x = torch.randn((1,10,21,21))
print(x)
print(x.shape)
arr = x.numpy()
arr_ = np.squeeze(arr)
print(arr_.shape)

print(arr_[0].shape)
plt.imshow(np.transpose(arr_[0]))
plt.show()