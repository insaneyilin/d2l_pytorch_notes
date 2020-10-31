import os
import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append("..") 
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.__version__)
print(device)


# 参数 aug 表示不同的增强方法
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5, title=None):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale, title)


# load an image
d2l.set_figsize()
img = Image.open('./img/cat1.jpg')
d2l.plt.imshow(img)

# Flip
apply(img, torchvision.transforms.RandomHorizontalFlip(), title='H Flip')
apply(img, torchvision.transforms.RandomVerticalFlip(), title='V Flip')

# Random resize & crop
shape_aug = torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug, title='Random resize & crop')

# Color Jittering
apply(img, torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0))
apply(img, torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5))
apply(img, torchvision.transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0))
color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)

# Composing multiple augmentations
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)

# 使用 CIFAR10 数据集
all_imges = torchvision.datasets.CIFAR10(train=True, root="./CIFAR", download=True)
# all_imges的每一个元素都是(image, label)
d2l.show_images([all_imges[i][0] for i in range(32)], 4, 8, scale=0.8);

# 训练集合上使用 flip
flip_aug = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])

# 测试集上不使用 augmentation
no_aug = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])

# 检查操作系统平台
num_workers = 0 if sys.platform.startswith('win32') else 4
# 读取数据集，返回一个 DataLoader
def load_cifar10(is_train, augs, batch_size, root="./CIFAR"):
    dataset = torchvision.datasets.CIFAR10(root=root, train=is_train, transform=augs, download=True)
    # is_train: 训练集的时候进行 shuffle
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)

# 带数据增强的训练
def train_with_data_aug(train_augs, test_augs, lr=0.001):
    batch_size, net = 128, d2l.resnet18(10)  # 使用 resnet-18
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()  # 交叉熵损失用于分类
    # 分别读取训练集和测试集
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    d2l.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs=10)

# show all figures
#d2l.plt.show()

# 使用数据增强进行训练；测试集不用数据增强
train_with_data_aug(flip_aug, no_aug)

