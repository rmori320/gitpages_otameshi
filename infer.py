# -*- coding: utf-8 -*-
"""
8. 学習済みモデルで推論する
snapshotでも、npzでも使ってやる

"""
import chainer
from chainer import serializers
from chainer.cuda import to_cpu
import chainer.links as L
import chainer.functions as F
from flask import Flask

app = Flask(__name__)


class MLP(chainer.Chain):

    def __init__(self, n_mid_units=100, n_out=10):
        super(MLP, self).__init__()

        # パラメータを持つ層の登録
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(n_mid_units, n_mid_units)
            self.l3 = L.Linear(n_mid_units, n_out)

    def __call__(self, x):
        # データを受け取った際のforward計算を書く
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
 
    
def infer(x):
    ##推論用のネットワーク  
    infer_net = MLP()
    serializers.load_npz(
        'mnist_result/snapshot_epoch-10',
        infer_net, path='updater/model:main/predictor/')
    
    ####################################
    #import numpy as np
    #from PIL import Image
    
    #img = Image.open('uploads/mnist_test.png')
    #x = np.array(img, dtype=np.float32)
    ######################################
    
    x = infer_net.xp.asarray(x[None, ...])
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        y = infer_net(x)
    y = to_cpu(y.array)
    
    #print('予測ラベル:', y.argmax(axis=1)[0])
    
    infer_num = y.argmax(axis=1)[0]
    
    return infer_num
