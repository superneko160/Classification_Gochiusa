import keras
import sys, os
import codeinfo
import numpy as np
from PIL import Image, ImageFile
from keras.models import load_model

ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_image(path):
    img = Image.open(path)
    img = img.convert("RGB")
    img = img.resize((codeinfo.IMAGE_SIZE, codeinfo.IMAGE_SIZE)) # 学習時にリサイズしたのと同じサイズに
    img = np.asarray(img)  # 画像をnumpy配列の形式に
    img = img / 255.0
    return img

def main():
    img = load_image(codeinfo.QUESTION_PIC)
    model = load_model(codeinfo.KERAS_PARAM)
    pred = model.predict(np.array([img]))
    print(pred)  # 精度の表示
    predlabel = np.argmax(pred, axis=1)
    if predlabel == 0:
        print("チノちゃん")
    elif predlabel == 1:
        print("ココアちゃん")

if __name__ == "__main__":
    main()