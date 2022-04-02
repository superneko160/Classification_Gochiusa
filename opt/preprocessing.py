import os, glob
import numpy as np
import codeinfo
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

classes = codeinfo.CLASSES
image_size = codeinfo.IMAGE_SIZE 
num_testdata = 25

X_train = []
X_test = []
y_train = []
y_test = []
# 前処理
for index, classlabel in enumerate(classes):
    photos_dir = "opt/data/" + classlabel
    files = glob.glob(photos_dir + "/*.jpg")
    for i, file in enumerate(files):
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size)) # 64*64のサイズに
        data = np.asarray(image)
        if i < num_testdata:
            X_test.append(data)
            y_test.append(index)
        else:
            # 画像を5度ずつ回転
            for angle in range(-20, 20, 5):
                img_r = image.rotate(angle)
                data = np.asarray(img_r)
                X_train.append(data)
                y_train.append(index)
                # さらに左右反転した画像も追加
                img_trains = img_r.transpose(Image.FLIP_LEFT_RIGHT)
                data = np.asarray(img_trains)
                X_train.append(data)
                y_train.append(index)
# ndarray型に変換
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
# NumPy配列（ndarray）をバイナリ形式で保存
xy = (X_train, X_test, y_train, y_test)
np.save("opt/data/dog_cat.npy", xy)