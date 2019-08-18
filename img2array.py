# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 16:18:41 2019

@author: Ryosuke Mori
"""

import numpy as np
from PIL import Image

img = Image.open('mnist_test.png')
arr = np.array(img, dtype=np.float32)