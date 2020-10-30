import os
import time
import torch
from torch import nn, optim
import torch.nn.functional as F

import sys
import d2lzh_pytorch as d2l

# check device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.__version__)
print(device)


# 残差块
# ResNet沿用了VGG全 3x3 卷积层的设计；
# 残差块里首先有2个有相同输出通道数的 3x3 卷积层(下面的 conv1 和 conv2)；
# 每个卷积层后接一个BN层和ReLU激活函数；
# 然后我们将输入跳过这两个卷积运算后直接加在最后的ReLU激活函数前；
# 这样的设计要求两个卷积层的输出与输入形状一样，从而可以相加；
# 如果想改变通道数，就需要引入一个额外的 1x1 卷积层来将输入变换成需要的形状后再做相加运算。(use_1x1conv)
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        # Y 学习残差, 即 f(X) - X
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        # 残差 + 原输入
        return F.relu(Y + X)

# 输入和输出形状一致的情况
blk = Residual(3, 3)
X = torch.rand((4, 3, 6, 6))
print(blk(X).shape) # torch.Size([4, 3, 6, 6])

# 增加输出通道数的同时减半输出的高和宽
# （1x1 conv 可以改变输出通道数，起到升维、降维的作用）
blk = Residual(3, 6, use_1x1conv=True, stride=2)
print(blk(X).shape) # torch.Size([4, 6, 3, 3])


# 构建残差块
def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


# resnet-18
def make_resnet18():
    net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # 每个 resnet block 使用两个残差块
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))

    # 全局平均池化 + FC
    # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
    net.add_module("global_avg_pool", d2l.GlobalAvgPool2d()) 
    net.add_module("fc", nn.Sequential(d2l.FlattenLayer(), nn.Linear(512, 10)))

    return net

net = make_resnet18()

# 逐层查看输出
X = torch.rand((1, 1, 224, 224))
for name, layer in net.named_children():
    X = layer(X)
    print(name, ' output shape:\t', X.shape)

# 测试 resnet 训练
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

