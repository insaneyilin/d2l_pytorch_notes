import os
import time
import torch
from torch import nn, optim
import torch.nn.functional as F

import sys
sys.path.append("..")  # to find d2lzh_pytorch
import d2lzh_pytorch as d2l

# check device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.__version__)
print(device)


# Inception block
# Inception块相当于一个有4条线路的子网络。它通过不同窗口形状的卷积层和最大池化层来并行抽取信息,
# 并使用1×1卷积层减少通道数从而降低模型复杂度
class Inception(nn.Module):
    # c1 - c4为每条线路里的层的输出通道数
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # 线路1，单 1x1 卷积层
        self.path1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
        # 线路2，1x1 卷积层后接 3x3 卷积层
        self.path2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
        self.path2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1 卷积层后接 5x5 卷积层
        self.path3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.path3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3 最大池化层后接 1x1 卷积层
        self.path4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.path4_2 = nn.Conv2d(in_c, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.path1_1(x))
        p2 = F.relu(self.path2_2(F.relu(self.path2_1(x))))
        p3 = F.relu(self.path3_2(F.relu(self.path3_1(x))))
        p4 = F.relu(self.path4_2(self.path4_1(x)))
        # 在通道维上连结输出, nchw, dim=1 表示在 c 上连接
        return torch.cat((p1, p2, p3, p4), dim=1)


# GoogleNet blocks
# GoogLeNet将多个设计精细的Inception块和其他层串联起来。
# 其中Inception块的通道数分配之比是在ImageNet数据集上通过大量的实验得来的
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   d2l.GlobalAvgPool2d())

# 查看每一层输出的形状
net = nn.Sequential(b1, b2, b3, b4, b5, d2l.FlattenLayer(), nn.Linear(1024, 10))
X = torch.rand(1, 1, 96, 96)
for blk in net.children():
    X = blk(X)
    print('output shape: ', X.shape)

batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

