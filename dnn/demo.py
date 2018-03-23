# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from PIL import Image
import h5py
import dnn
from dnn_app_utils import load_data

# 导入数据
train_x_origin, train_y, test_x_origin, test_y, classes = load_data()

# 图像数据预处理
# print(train_x_origin.shape)
train_x_flatten = train_x_origin.reshape(train_x_origin.shape[0], -1).T
test_x_flatten = test_x_origin.reshape(test_x_origin.shape[0], -1).T
train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

# 设置网络并运行
layers_dims = [test_x.shape[0], 20, 7, 5, 1]

parameters = dnn.L_layer_model(train_x, train_y, layers_dims, learning_rate=0.0075, num_iterations=2500,print_cost=True)
