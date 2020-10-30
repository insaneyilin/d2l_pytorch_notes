import os
import time
import torch
from torch import nn, optim

import sys
import d2lzh_pytorch as d2l


# check device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.__version__)
print(device)


# VGG block
def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    # 采用堆积的小卷积核代替单一的大卷积核；在相同感受野的情况下提升了网络的深度
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    # 2x2 max pooling, 宽高减半
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*blk)


# conv_arch 里的元素: (卷积层数量, in_channels, out_channels)
# 这里是 8 个卷积层
conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
fc_features = 512 * 7 * 7    # 根据卷积层的输出算出来的
fc_hidden_units = 4096       # 任意


# make VGG net
def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    # 卷积层部分，根据 conv_arch 堆叠 VGG blocks
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        net.add_module("vgg_block_" + str(i+1),
                       vgg_block(num_convs, in_channels, out_channels))
    # 全连接层部分, 3 个 fc
    net.add_module("fc", nn.Sequential(d2l.FlattenLayer(),
                                       nn.Linear(fc_features, fc_hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(fc_hidden_units, fc_hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(fc_hidden_units, 10)
                                      ))
    return net


# 构建一个 VGG-11 （8 个卷积层 + 3 个全连接层）
net = vgg(conv_arch, fc_features, fc_hidden_units)

# 构造一个输入，看下 VGG 中间层的形状
X = torch.rand(1, 1, 224, 224)

for name, blk in net.named_children():
    X = blk(X)  # 逐层进行 forward
    print(name, 'output shape: ', X.shape)

# 为了演示，用 ratio 减少通道数
ratio = 8

small_conv_arch = [(1, 1, 64//ratio), (1, 64//ratio, 128//ratio), (2, 128//ratio, 256//ratio),
                   (2, 256//ratio, 512//ratio), (2, 512//ratio, 512//ratio)]
net = vgg(small_conv_arch, fc_features // ratio, fc_hidden_units // ratio)
print(net)

batch_size = 64
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

# 测试 VGG 训练
lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

