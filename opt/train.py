from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import RMSprop
from keras.utils import np_utils
import keras
import codeinfo
import numpy as np

classes = codeinfo.CLASSES
num_classes = codeinfo.CLASSES_LENGTH

def load_data():
    X_train,X_test,y_train,y_test = np.load("opt/data/dog_cat.npy", allow_pickle=True)
    # 入力データの各画素数を0〜1のは2で正規化
    X_train = X_train.astype("float") / 255
    X_test = X_test.astype("float") / 255
    # to_categorical()にてラベルをone hot vector化
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    return X_train, y_train, X_test, y_test

def train(X, y, X_test, y_test):
    model = Sequential()

    model.add(Conv2D(32, (3,3), padding='same', input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.45))
    model.add(Dense(codeinfo.CLASSES_LENGTH))  # nクラスに分類
    model.add(Activation('softmax'))  # softmax関数

    # 最適化アルゴリズムにRMSpropを採用
    opt = RMSprop(lr=0.00005, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(X, y, batch_size=28, epochs=40)
    # HDF5ファイルにKerasのモデルを保存
    model.save('opt/data/cnn.h5')
    return model

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()
    model = train(X_train, y_train, X_test, y_test)
