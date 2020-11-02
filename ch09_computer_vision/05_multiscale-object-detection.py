from PIL import Image
import numpy as np
import torch

import sys
sys.path.append("..") 
import d2lzh_pytorch as d2l
print(torch.__version__)

img = Image.open('./img/catdog.jpg')
w, h = img.size
print(w, h)

# 特征图: 卷积神经网络的二维数组输出
# 我们可以通过定义特征图的形状来确定任一图像上均匀采样的锚框中心 
# 下面定义 `display_anchors`函数。我们在特征图 `fmap` 上以每个单元（像素）为中心生成锚框`anchors`。
# 由于锚框 `anchors` 中 x 和 y 轴的坐标值分别已除以特征图 `fmap` 的宽和高，这些值域在0和1之间的值表达了锚框在特征图中的相对位置。
# 由于锚框 `anchors` 的中心遍布特征图 `fmap` 上的所有单元，`anchors` 的中心在任一图像的空间相对位置一定是均匀分布的。
# 具体来说，当特征图的宽和高分别设为 `fmap_w` 和 `fmap_h` 时，该函数将在任一图像上均匀采样 `fmap_h` 行 `fmap_w` 列个像素，并分别以它们为中心生成大小为 `s`（假设列表 `s` 长度为1）的不同宽高比（ratios）的锚框。

d2l.set_figsize()

def display_anchors(fmap_w, fmap_h, s):
    # 前两维的取值不影响输出结果
    fmap = torch.zeros((1, 10, fmap_h, fmap_w), dtype=torch.float32)

    # 平移所有锚框使均匀分布在图片上
    offset_x, offset_y = 1.0/fmap_w, 1.0/fmap_h
    anchors = d2l.MultiBoxPrior(fmap, sizes=s, ratios=[1, 2, 0.5]) + \
        torch.tensor([offset_x/2, offset_y/2, offset_x/2, offset_y/2])

    bbox_scale = torch.tensor([[w, h, w, h]], dtype=torch.float32)
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)

# 先关注小目标的检测。为了在显示时更容易分辨，这里令不同中心的锚框不重合：
# 设锚框大小为0.15，特征图的高和宽分别为2和4。可以看出，图像上2行4列的锚框中心分布均匀。
d2l.plt.figure()
display_anchors(fmap_w=4, fmap_h=2, s=[0.15])

# 将特征图的高和宽分别减半，并用更大的锚框检测更大的目标。当锚框大小设0.4时，有些锚框的区域有重合。
d2l.plt.figure()
display_anchors(fmap_w=2, fmap_h=1, s=[0.4])

# 将特征图的宽进一步减半至1，并将锚框大小增至0.8。此时锚框中心即图像中心。
d2l.plt.figure()
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])

# 既然我们已在多个尺度上生成了不同大小的锚框，相应地，我们需要在不同尺度下检测不同大小的目标。
# 思路：c_i 张特征图，特征图在相同空间位置的 c_i 个单元在输入图像上的感受野相同，并表征了同一感受野内的输入图像信息。
# 本质上，我们用输入图像在某个感受野区域内的信息来预测输入图像上与该区域位置相近的锚框的类别和偏移量。
# 当不同层的特征图在输入图像上分别拥有不同大小的感受野时，它们将分别用来检测不同大小的目标。例如，我们可以通过设计网络，令较接近输出层的特征图中每个单元拥有更广阔的感受野，从而检测输入图像中更大尺寸的目标。

d2l.plt.show()

