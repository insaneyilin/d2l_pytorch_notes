# -*- coding: utf-8 -*-

from PIL import Image

import sys
sys.path.append("..") 
import d2lzh_pytorch as d2l

d2l.set_figsize()
img = Image.open('img/catdog.jpg')
d2l.plt.imshow(img)

# [left, top, right, bottom]
dog_bbox, cat_bbox = [60, 45, 378, 516], [400, 112, 655, 493]

def bbox_to_rect(bbox, color):
    # (x, y, width, height)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)

fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))

d2l.plt.show()

