import numpy as np
import pandas as pd
import os
# import cv2
from keras.preprocessing import image
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold

from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, GlobalMaxPooling2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.applications import *
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.preprocessing.image import load_img, img_to_array

IMG_W, IMG_H = (448, 448)
N_CLASSES = 100

def collect_train_data():
    train_path = './input/train'
    x_train = []
    y_train = []

    for _, dirs, _ in os.walk(train_path):
        for claz in dirs:
            for _, _, files in os.walk(train_path + '/' + claz):
                for file in files:
                    # x_train.append(cv2.resize(cv2.imread(train_path + '/' + claz + '/' + file), (IMG_W, IMG_H)))
                    img = image.load_img(train_path + '/' + claz + '/' + file, target_size=(IMG_W, IMG_H))
                    img = image.img_to_array(img)
                    x_train.append(img)
                    targets = np.zeros(100)
                    targets[int(claz) - 1] = 1
                    y_train.append(targets)

    x_train = np.array(x_train, np.float32)
    y_train = np.array(y_train, np.uint8)

    print(x_train.shape)
    print(y_train.shape)
    #val data
    # val_path = './input/val'
    # x_val = []
    # y_val = []
    #
    # for _, dirs, _ in os.walk(val_path):
    #     for claz in dirs:
    #         for _, _, files in os.walk(val_path + '/' + claz):
    #             for file in files:
    #                 # x_train.append(cv2.resize(cv2.imread(train_path + '/' + claz + '/' + file), (IMG_W, IMG_H)))
    #                 img = image.load_img(val_path + '/' + claz + '/' + file, target_size=(IMG_W, IMG_H))
    #                 img = image.img_to_array(img)
    #                 x_val.append(img)
    #                 targets = np.zeros(100)
    #                 targets[int(claz) - 1] = 1
    #                 y_val.append(targets)
    #
    # x_val = np.array(x_val, np.float32)
    # y_val = np.array(y_val, np.uint8)
    #
    # print(x_val.shape)
    # print(y_val.shape)

    return x_train, y_train #, x_val,y_val


def build_and_compile_model(lr=0.0001):
    # base_model = InceptionV3(input_shape=(IMG_W, IMG_H, 3), include_top=False, weights='imagenet', pooling='avg')
    # base_model = Xception(input_shape=(IMG_W, IMG_H, 3), include_top=False, weights='imagenet', pooling='avg')
    base_model = DenseNet201(input_shape=(IMG_W, IMG_H, 3), include_top=False, weights='imagenet', pooling='avg',classes=N_CLASSES)
    x = base_model.output
    x = GlobalMaxPooling2D(input_shape=(7,7,1024),data_format='channels_last')(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(100, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr), metrics=['accuracy'])

    return model


def train_model(model, x_train, y_train,model_sn=0, batch_size=8, epochs=35):
    # datagen = ImageDataGenerator( horizontal_flip=True,
    #                               vertical_flip=True)
    # datagen = ImageDataGenerator()
    datagen = ImageDataGenerator(
                 rotation_range = 20,
                 width_shift_range = 0.2,
                 height_shift_range = 0.2,
                # rescale = 1./255,
                shear_range = 0.2,
        zoom_range=0.2
        # horizontal_flip=True,
        #         fill_mode = 'nearest'
    )

    # valid_datagen = ImageDataGenerator()

    weights = os.path.join('weights/', "weights_{}.h5".format(model_sn))

    callbacks = [#EarlyStopping(monitor='val_loss', patience=5, verbose=0),
                 ModelCheckpoint(weights, monitor='acc', save_best_only=True, verbose=0),
                 ReduceLROnPlateau(monitor='acc', factor=0.1, patience=2, verbose=0, mode='auto', epsilon=0.0001,
                                   cooldown=0, min_lr=0)]

    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(x_train) // batch_size,
                        # validation_data=valid_datagen.flow(x_valid, y_valid, batch_size=batch_size),
                        # validation_steps=len(x_valid) / batch_size,
                        callbacks=callbacks,
                        epochs=epochs,
                        verbose=1)


def train_model_on_k_fold(get_model, train_model, X, y, n_folds=5):
    # sss = StratifiedShuffleSplit(n_splits=n_folds,test_size=0.16,random_state=42)
    # sss = StratifiedKFold(n_splits=n_folds,random_state=42,shuffle=True)
    n_models = 0
    models = []
    for k in range(n_folds):
        print("Building model...")
        models.append(get_model())
        print("Trainning model...")
        train_model(models[len(models) - 1], X, y, model_sn=n_models)

        n_models += 1
    # for k,(train_index, test_index) in enumerate(sss.split(X, y)):
    #     print(k)
    #     print("Using {} for training and {} for validation".format(len(train_index), len(test_index)))
    #     x_train, x_valid = X[train_index], X[test_index]
    #     y_train, y_valid = y[train_index], y[test_index]
    #
    #     print("Building model...")
    #     models.append(get_model())
    #     print("Trainning model...")
    #     train_model(models[len(models) - 1], x_train, y_train, x_valid, y_valid, model_sn=n_models)
    #
    #     n_models += 1

    return models


def generate_result_and_write_to_file(models, filename='result.csv'):
    test_path = './input/test'

    X_test = []
    test_files = []

    for _, _, files in os.walk(test_path):
        for file in files:
            test_files.append(file)
            # img = cv2.resize(cv2.imread(test_path + '/' + file), (IMG_W, IMG_H))
            img = image.load_img(test_path + '/' + file, target_size=(IMG_W, IMG_H))
            img = image.img_to_array(img)
            X_test.append(img)

    X_test = np.array(X_test)

    tmp = np.zeros((X_test.shape[0], N_CLASSES))
    for model in models:
        tmp += model.predict(X_test)

    ans = np.argmax(tmp, axis=1) + 1

    result = dict(zip(test_files, ans))

    filelist = open('./input/test.txt')
    f = open('./result/' + filename, 'w')
    a = []
    for filename in filelist:
        filename = filename.strip()
        a.append([filename, result[filename]])
    filelist.close()
    df = pd.DataFrame(a)
    df.to_csv('./result/result.csv', header=None, index=None, sep=' ')
    return df

# X, y = collect_train_data()
# models = train_model_on_k_fold(build_and_compile_model, train_model, X, y)

def load_models(path='./weights'):
    models = []
    for _, _, files in os.walk(path):
        for file in files:
            model = build_and_compile_model()
            model.load_weights(path + '/' + file)
            models.append(model)
    return models

models = load_models()
generate_result_and_write_to_file(models)