# 导入包，后续进行训练、测试，输出分类正确率
# -*- encoding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn import functional as F
import warnings
from sklearn.naive_bayes import GaussianNB
import sklearn

batch_size = 50
embedding_dim = 300  # 词向量维度
filter_num = 100  # 卷积器数量
sentence_max_size = 40  # 句子最大长度
label_size = 6  # 分类输出数量
kernel_list = [3, 4, 5]  # 3种卷积器大小，每种100个
num_epoch = 200  # 设置迭代次数
lr = 0.02  # 学习率

train_set_name = "save4/train_set" + str(1) + ".pth"
train_label_name = "save4/train_label" + str(1) + ".pth"
val_set_name = "save4/val_set" + str(1) + ".pth"
val_label_name = "save4/val_label" + str(1) + ".pth"
test_set_name = "save4/test_set" + str(1) + ".pth"
test_label_name = "save4/test_label" + str(1) + ".pth"

train_set = torch.load(train_set_name)  # 训练数据
train_label = torch.load(train_label_name)  # 训练数据标签
val_set = torch.load(val_set_name)  # 验证数据
val_label = torch.load(val_label_name)  # 验证数据标签
test_set = torch.load(test_set_name)  # 测试数据
test_label = torch.load(test_label_name)  # 测试集标签

print(train_set.size())
print(train_label.size())
print(test_set.size())
print(test_label.size())
print(val_set.size())
print(val_label.size())

train_set = train_set.numpy()  # 训练数据
train_label = train_label.numpy()  # 训练数据标签
test_set = test_set.numpy()  # 测试数据
test_label = test_label.numpy()  # 测试集标签

print(train_set.shape)
print(train_set.type)
gNB = GaussianNB()
gNB.fit(train_set, train_label)  # 训练模型
gNB.score(test_set, test_label)  # 测试

# train_set = TensorDataset(train_set, train_label)  # 训练数据装成TensorDataset
# val_set = TensorDataset(val_set, val_label)  # 验证数据装成TensorDataset
# test_set = TensorDataset(test_set, test_label)  # 测试数据装成TensorDataset
#
# Train_DataLoader = DataLoader(train_set, batch_size=batch_size,
#                               shuffle=False)  # 使用DataLoader组织数据, 设置batch_size, 打乱数据
# Val_DataLoader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
# Test_DataLoader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

#print(train_set.type)
#print(train_set.shape)


