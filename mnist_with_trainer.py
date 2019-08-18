# -*- coding: utf-8 -*-

""""
chainerの便利な機能を使わないでmnist推定
"""


"""
1.データセットの準備
"""
from chainer.datasets import mnist 
# 訓練用60000枚、テスト用10000枚
# ndim=1 だから(784,)の１次元の配列で与えられる
train_val, test = mnist.get_mnist(withlabel=True, ndim = 1)

# 試しにmatplotlibで描画
import matplotlib.pyplot as plt

x, t = train_val[0]
plt.imshow(x.reshape(28,28), cmap='gray')
plt.axis('off')
plt.show()
print('label;', t)


# validation用のデータセットを作成
from chainer.datasets import split_dataset_random
# 訓練用の60000枚をtrainとvalidationに分ける
train, valid = split_dataset_random(train_val, 50000, seed=0)


"""
2.Iteratorの作成
ミニバッチ学習させるにあたって、バッチサイズ分だけデータとラベルを束ねる作業が面倒
そこで、データセットから決まった数のデータとラベルを取得して、ミニバッチを作ってくれるIteratorを使う
"""

from chainer import iterators
batchsize = 128

train_iter = iterators.SerialIterator(train, batchsize)
valid_iter = iterators.SerialIterator(valid, batchsize, repeat=False, shuffle=False)
test_iter = iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)


"""
3.ネットワークの定義
Chainはパラメータを持つ層Linkをまとめておくためのクラス
(パラメータを持つ→更新される必要がある、更新するのはoptimizer)
Chainクラスはto_gpuメソッドを持つ
"""

import chainer
import chainer.links as L
import chainer.functions as F

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

gpu_id = -1  # CPUを用いる場合は、この値を-1にしてください

net = MLP()

if gpu_id >= 0:
    net.to_gpu(gpu_id)



import random
import numpy
def reset_seed(seed=0):
    random.seed(seed)
    numpy.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)


"""
4.最適化手法の選択
上記の「パラメータを持つ層」を更新していく
"""

"""
5.学習する
# ---------- 学習の1イテレーション ----------
train_batch = train_iter.next()
x, t = concat_examples(train_batch, gpu_id)
# 予測値の計算
y = net(x)
# ロスの計算
loss = F.softmax_cross_entropy(y, t)
# 勾配の計算
net.cleargrads()
loss.backward()
# パラメータの更新
optimizer.update()


以上を、updater を使って簡潔に書く！
"""

from chainer import training,optimizers
gpu_id = -1  

# ネットワークをClassifierで包んで、ロスの計算などをモデルに含める
net = L.Classifier(net)
# 最適化手法の選択
optimizer = optimizers.SGD(lr=0.01).setup(net)
# UpdaterにIteratorとOptimizerを渡す
updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)


"""
6. Trainerの準備

学習のループを簡潔化しているのがupdater
TrainerはさらにUpdaterを受け取って学習全体の管理を行う機能がある(extensionsたち)
便利な機能を使うためにUpdaterをTrainerに受け渡す
"""

max_epoch = 10

# TrainerにUpdaterを渡す
trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='mnist_result')

from chainer.training import extensions

trainer.extend(extensions.LogReport())
trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
trainer.extend(extensions.Evaluator(valid_iter, net, device=gpu_id), name='val')
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'elapsed_time']))
trainer.extend(extensions.PlotReport(['l1/W/data/std'], x_key='epoch', file_name='std.png'))
trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
trainer.extend(extensions.dump_graph('main/loss'))

trainer.run()


"""
7.テストデータで検証する
"""

test_evaluator = extensions.Evaluator(test_iter, net, device=gpu_id)
results = test_evaluator()
print('Test accuracy:', results['main/accuracy'])



"""
8. 学習済みモデルで推論する
snapshotでも、npzでも使ってやる

"""
from chainer import serializers
from chainer.cuda import to_cpu


reset_seed(0)

infer_net = MLP()
serializers.load_npz(
    'mnist_result/snapshot_epoch-10',
    infer_net, path='updater/model:main/predictor/')

if gpu_id >= 0:
    infer_net.to_gpu(gpu_id)

x, t = test[0]
plt.imshow(x.reshape(28, 28), cmap='gray')
plt.show()

x = infer_net.xp.asarray(x[None, ...])
with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
    y = infer_net(x)
y = to_cpu(y.array)
    
print('予測ラベル:', y.argmax(axis=1)[0])
