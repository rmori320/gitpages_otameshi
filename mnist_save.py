# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 14:01:45 2019

@author: Ryosuke Mori
"""

import chainer

from PIL import Image

train, test = chainer.datasets.get_mnist()
data = test[9][0]

img = Image.new("L", (28,28))
pix = img.load()

for i in range(28):
    for j in range(28):
        pix[i,j] = int(data[i+j*28]*256)
        
#img2 = img.resize((280,280))

img.save("mnist_test.png")