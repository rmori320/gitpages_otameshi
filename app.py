# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 14:38:37 2019

@author: Ryosuke Mori
"""

import os
import io
import time
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug import secure_filename
from PIL import Image
import infer



app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'PNG', 'JPG'])
IMAGE_WIDTH = 640
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


## localhostにアクセスしたら、ただ index.htmlを返す
@app.route('/')
def index():
    return render_template('index.html')

## 返した index.htmlから、画像が postされてくるようになってる
## index.html の <form>の部分を見ればわかる
## methodがpost で /sendに送られてくると記述されている
@app.route('/send', methods=['GET', 'POST'])
def send():
    if request.method == 'POST':
        img_file = request.files['img_file']
        img = Image.open(img_file)
        
        img_url = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
        img.save(img_url)
        
        img_array = np.array(img, dtype=np.float32)
        ans = infer.infer(img_array)
        return render_template('index.html', message = ans, img_url = img_url)
        
    else:
        return redirect(url_for('index'))
    


"""
        # 変なファイル弾き
        if img_file and allowed_file(img_file.filename):
            filename = secure_filename(img_file.filename)
        else:
            return ''' <p>許可されていない拡張子です</p> '''

"""


##send_from_directory名前の通り、ディレクトリ内にある画像をwebブラウザに送る
##これがないと、localhostの画面に画像が表示されなかった

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



if __name__ == '__main__':
    app.debug = True
    app.run()