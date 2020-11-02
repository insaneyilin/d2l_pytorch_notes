from PIL import Image
import numpy as np
import math
import torch

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
print(torch.__version__)


d2l.set_figsize()
img = Image.open('./img/catdog.jpg')
w, h = img.size
print("w = %d, h = %d" % (w, h))

d2l.plt.figure()
d2l.plt.imshow(img)

# Anchor: 以每个像素为中心生成多个大小和宽高比（aspect ratio）不同的边界框
def MultiBoxPrior(feature_map, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5]):
    """
    # 这里 anchor 表示成 (xmin, ymin, xmax, ymax).
    Args:
        feature_map: torch tensor, Shape: [N, C, H, W].
        sizes: List of sizes (0~1) of generated MultiBoxPriores.
        ratios: List of aspect ratios (non-negative) of generated MultiBoxPriores.
    Returns:
        anchors of shape (1, num_anchors, 4). 由于batch里每个都一样, 所以第一维为1
    """
    # 代码细节：如果用所有 size 和 ratio 的组合，计算复杂度太高；
    # 通常只对包含 s1 或 r1 的大小与宽高比的组合感兴趣
    pairs = []  # pair of (size, sqrt(ratio))
    # 只包含 s1 的组合
    for r in ratios:
        pairs.append([sizes[0], math.sqrt(r)])
    # 只包含 r1 的组合
    for s in sizes[1:]:
        pairs.append([s, math.sqrt(ratios[0])])

    pairs = np.array(pairs)

    # anchor 宽的缩放系数 list, anchor 的宽 = w * s * sqrt(r), w 是输入图像的宽
    ss1 = pairs[:, 0] * pairs[:, 1]  # size * sqrt(ration)
    print('\nss1: ')
    print(ss1)
    # anchor 高的缩放系数 list，anchor 的高 = h * s / sqrt(r), h 是输入图像的高
    ss2 = pairs[:, 0] / pairs[:, 1]  # size / sqrt(ration)
    print('\nss2: ')
    print(ss2)

    # axis = 1 表示在"列"维度上进行拼接
    # ss1, ss2 都是列向量，这里得到一个列数为 4 的矩阵
    base_anchors = np.stack([-ss1, -ss2, ss1, ss2], axis=1) / 2
    print('\nbase_anchors: ')
    print(base_anchors)

    h, w = feature_map.shape[-2:]
    # 生成归一化的 x-y 网格点
    shifts_x = np.arange(0, w) / w
    shifts_y = np.arange(0, h) / h
    print('\nshifts_x: ', shifts_x)
    print('\nshifts_y: ', shifts_y)
    shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    # shifts 是一个列数为 4 的矩阵，每一行是一个网格点坐标重复两次
    shifts = np.stack((shift_x, shift_y, shift_x, shift_y), axis=1)

    # TODO: 这里细节还没搞清楚
    anchors = shifts.reshape((-1, 1, 4)) + base_anchors.reshape((1, -1, 4))

    return torch.tensor(anchors, dtype=torch.float32).view(1, -1, 4)


# 构造输入数据
X = torch.Tensor(1, 3, h, w)
Y = MultiBoxPrior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print('\nY.shape: ', Y.shape)

boxes = Y.reshape((h, w, 5, 4))
print(boxes[250, 250, 0, :])  # * torch.tensor([w, h, w, h], dtype=torch.float32)
# 第一个size和ratio分别为0.75和1, 则宽高均为0.75 = 0.7184 + 0.0316 = 0.8206 - 0.0706

d2l.set_figsize()
d2l.plt.figure()
fig = d2l.plt.imshow(img)
bbox_scale = torch.tensor([[w, h, w, h]], dtype=torch.float32)
d2l.show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.75, r=2', 's=0.55, r=0.5', 's=0.5, r=1', 's=0.25, r=1'])

# 如何刻画 anchor 和 gt bbox 之间的相似度？
# Jaccard系数（Jaccard index）, 即 IoU

# 如何对 anchor 进行标注？
# 将每个锚框视为一个训练样本，为每个锚框标注两类标签：
# (1) anchor 包含目标的类别；(2) gt bbox 相对于 anchor 的 偏移量(offset)
# 在目标检测时，我们首先生成多个锚框，然后为每个锚框预测类别以及偏移量，
# 接着根据预测的偏移量调整锚框位置从而得到预测边界框，最后筛选需要输出的预测边界框。

# 如何为锚框分配与其相似的真实边界框？
# 利用 IoU 建立匹配 similarity/cost 矩阵

# 由于数据集中各个框的位置和大小各异，因此这些相对位置和相对大小通常需要一些特殊变换，才能使偏移量的分布更均匀从而更容易拟合
# “边框回归变换”

# anchor 中的正负样本
# 如果一个锚框没有被分配真实边界框，我们只需将该锚框的类别设为背景。类别为背景的锚框通常被称为负类锚框，其余则被称为正类锚框。

# 一个具体的例子。我们为读取图像中的猫狗定义真实边界框，其中第一个元素为类别（0为狗，1为猫）
# 剩余4个元素分别为左上角的x和y轴坐标以及右下角的x和y轴坐标（值域在0到1之间）。
# 这里通过左上角和右下角的坐标构造了5个需要标注的锚框
bbox_scale = torch.tensor((w, h, w, h), dtype=torch.float32)
ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                            [1, 0.55, 0.2, 0.9, 0.88]])
anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])

d2l.plt.figure()
fig = d2l.plt.imshow(img)
d2l.show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
d2l.show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])

# 通过unsqueeze函数为锚框和真实边界框添加样本维
labels = d2l.MultiBoxTarget(anchors.unsqueeze(dim=0),
                        ground_truth.unsqueeze(dim=0))
# 返回的结果：
# bbox_offset: 每个锚框的标注偏移量，形状为(bn，锚框总数*4)
# bbox_mask: 形状同bbox_offset, 每个锚框的掩码, 一一对应上面的偏移量, 负类锚框(背景)对应的掩码均为0, 正类锚框的掩码均为1
# cls_labels: 每个锚框的标注类别, 其中0表示为背景, 形状为(bn，锚框总数)
print(labels[2])
# tensor([[0, 1, 2, 0, 2]]), 2分类，0 表示背景类

# 返回值的第二项为掩码（mask）变量，形状为(批量大小, 锚框个数的四倍)。
# 掩码变量中的元素与每个锚框的4个偏移量一一对应。 由于我们不关心对背景的检测，
# 有关负类的偏移量不应影响目标函数。
# 通过按元素乘法，掩码变量中的0可以在计算目标函数之前过滤掉负类的偏移量。
print(labels[1])

# 返回值的第一项是为每个锚框标注的四个偏移量，其中负类锚框的偏移量标注为0。
print(labels[0])

# 输出预测边界框
# 当锚框数量较多时，同一个目标上可能会输出较多相似的预测边界框。
# 使用非极大值抑制（non-maximum suppression，NMS）移除相似的预测边界框
# NMS 简单记忆：置信度排序，IoU 阈值

# NMS 的例子
anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                        [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = torch.tensor([0.0] * (4 * len(anchors)))
cls_probs = torch.tensor([[0., 0., 0., 0.,],  # 背景的预测概率
                          [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                          [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率
d2l.plt.figure()
fig = d2l.plt.imshow(img)
d2l.show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])

from collections import namedtuple
Pred_BB_Info = namedtuple("Pred_BB_Info", ["index", "class_id", "confidence", "xyxy"])

# 运行目标检测，NMS 阈值设置为 0.5
# 返回的结果的形状为(批量大小, 锚框个数, 6)
# 每一行的6个元素代表同一个预测边界框的输出信息
# 第一个元素是索引从0开始计数的预测类别（0为狗，1为猫），
# 其中-1表示背景或在非极大值抑制中被移除。第二个元素是预测边界框的置信度。
# 剩余的4个元素分别是预测边界框左上角的x和y轴坐标以及右下角的x和y轴坐标（值域在0到1之间）
output = d2l.MultiBoxDetection(
    cls_probs.unsqueeze(dim=0), offset_preds.unsqueeze(dim=0),
    anchors.unsqueeze(dim=0), nms_threshold=0.5)

# 移除掉类别为-1的预测边界框，并可视化非极大值抑制保留的结果
d2l.plt.figure()
fig = d2l.plt.imshow(img)
for i in output[0].detach().cpu().numpy():
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    d2l.show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)

# 实践中，我们可以在执行非极大值抑制前将置信度较低的预测边界框移除，从而减小非极大值抑制的计算量。
# 我们还可以筛选非极大值抑制的输出，例如，只保留其中置信度较高的结果作为最终输出。

d2l.plt.show()

