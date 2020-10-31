# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import os

import sys
sys.path.append("..") 
import d2lzh_pytorch as d2l

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.__version__)
print(device)

# hotdog dataset
data_dir = './hotdog'
print(os.listdir(data_dir))

# 使用 ImageFolder 实例来读取文件
train_imgs = ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = ImageFolder(os.path.join(data_dir, 'test'))

hotdogs = [train_imgs[i][0] for i in range(8)]  # 前 8 张正样本
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]  # 最后 8 张负样本
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4, title='hotdog dataset')

# 在使用预训练模型时，一定要和预训练时作同样的预处理
# 要仔细阅读 pretrained-models 的说明，看其是如何预处理的
# 指定RGB三个通道的均值和方差来将图像通道归一化
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# 训练时，先从图像中裁剪出随机大小和随机高宽比的一块随机区域，然后将该区域缩放为高和宽均为224像素的输入
train_augs = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
# 测试时，将图像的高和宽均缩放为256像素，然后从中裁剪出高和宽均为224像素的中心区域作为输入
test_augs = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        normalize
    ])

# 预训练模型
# 指定pretrained=True来自动下载并加载预训练的模型参数
pretrained_net = models.resnet18(pretrained=True)
print(pretrained_net)
# 查看最后的全连接层，它将ResNet最终的全局平均池化层输出变换成ImageNet数据集上1000类的输出。
print(pretrained_net.fc)
# 修改全连接层，修改为我们需要的类别
pretrained_net.fc = nn.Linear(512, 2)
print(pretrained_net.fc)

# 此时，pretrained_net的fc层就被随机初始化了，但是其他层依然保存着预训练得到的参数。
# 由于是在很大的ImageNet数据集上预训练的，所以参数已经足够好，因此一般只需使用较小的学习率来微调这些参数，
# 而fc中的随机初始化参数一般需要更大的学习率从头训练。
# 下面将fc的学习率设为已经预训练过的部分的10倍
# 获取 fc 层中每个参数的 id
output_params = list(map(id, pretrained_net.fc.parameters()))
# 先获取所有参数，然后只保留不在输出层(fc层)中的参数，即得到了 feature params
feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())

lr = 0.01  # 默认学习率是 0.01
# 注意这里对 fc 层使用的学习率是默认学习率的 10 倍
optimizer = optim.SGD([{'params': feature_params},
                       {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
                       lr=lr, weight_decay=0.001)

def train_fine_tuning(net, optimizer, batch_size=128, num_epochs=5):
    train_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'train'), transform=train_augs),
                            batch_size, shuffle=True)
    test_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'test'), transform=test_augs),
                           batch_size)
    loss = torch.nn.CrossEntropyLoss()
    d2l.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

# 使用预训练模型进行训练
print('\n Use pretrained_net')
train_fine_tuning(pretrained_net, optimizer)

# 对比一下不使用预训练模型，定义一个相同的模型，但将它的所有模型参数都初始化为随机值。
# 由于整个模型都需要从头训练，我们可以使用较大的学习率
scratch_net = models.resnet18(pretrained=False, num_classes=2)
lr = 0.1
optimizer = optim.SGD(scratch_net.parameters(), lr=lr, weight_decay=0.001)
print('\n Use scratch_net')
train_fine_tuning(scratch_net, optimizer)

# 结论：微调的模型因为参数初始值更好，往往在相同迭代周期下取得更高的精度。

# show all figures
#d2l.plt.show()

