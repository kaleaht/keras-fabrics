import os, sys
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, Add, Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.engine.topology import Layer
import keras
from keras.utils import plot_model

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

sys.path.append('../..')
from helpers import model_history_to_png, PlotLosses

import tensorflow as tf

from bicubic import Bicubic

print('Loading and preprocessing training and validation data...')

path = '../../local_data/'

imgs_train = np.load(path + 'parts_train.npy')
imgs_mask_train = np.load(path + 'parts_train_labels.npy')

imgs_train = imgs_train.astype('float32')
mean = np.mean(imgs_train)  # mean for data centering
std = np.std(imgs_train)  # std for data normalization

imgs_train -= mean
imgs_train /= std

imgs_mask_train = imgs_mask_train.astype('float32')
imgs_mask_train /= 255.  # scale masks to [0, 1]


imgs_validation_org = np.load(path + 'parts_validation.npy')
imgs_validation = imgs_validation_org.astype('float32')
imgs_mask_validation_org = np.load(
    path + 'parts_validation_labels.npy')
imgs_mask_validation = imgs_mask_validation_org.astype('float32')

imgs_validation -= mean
imgs_validation /= std

imgs_mask_validation /= 255.  # scale masks to [0, 1]

flip = np.flip(imgs_train, axis=2)
flip_mask = np.flip(imgs_mask_train, axis=2)

imgs_train = np.concatenate([imgs_train, flip])
imgs_mask_train = np.concatenate([imgs_mask_train, flip_mask])
print('-' * 30)

def get_u_net(img_shape, n_class, depth, n_layers=8, filter_shape=(3, 3), kernel_shape=(4, 4)):
    img_rows = img_shape[0]
    img_cols = img_shape[1]
    inputs = Input((img_rows, img_cols, n_class))

    # array for first half of convolutional layers.
    # They are stored so we can use them in the upsampling part.
    conv_layers = []

    input_h = inputs
    #conv = inputs
    for i in range(0, depth):
        n_layers *= 2
        conv = Conv2D(n_layers, filter_shape, activation='relu',
                      padding='same', use_bias=False)(input_h)
        # conv = Conv2D(n_layers, filter_shape, activation='relu',
                      # padding='same', use_bias=False)(conv)
        conv = BatchNormalization()(conv)
        conv_layers.insert(0, conv)
        input_h = MaxPooling2D(pool_size=(2, 2))(conv)

    n_layers *= 2
    conv = Conv2D(n_layers, filter_shape, activation='relu',
                  padding='same', use_bias=False)(input_h)
    # conv = Conv2D(n_layers, filter_shape, activation='relu',
                  # padding='same', use_bias=False)(conv)

    for j in range(0, depth):
        n_layers = int(n_layers / 2)

        # Bicubic interpolation
        size = [int(conv_layers[j].shape[1]), int(conv_layers[j].shape[2])]
        print(size)
        conv_trans = Bicubic(size)(conv)

        conv_trans = Conv2D(n_layers, filter_shape,
                      activation='relu', padding='same', use_bias=False)(conv_trans)
        # conv_trans = Conv2DTranspose(
            # n_layers, kernel_shape, strides=(2, 2), padding='same')(conv)
        up = keras.layers.add([conv_trans, conv_layers[j]])
        conv = Conv2D(n_layers, filter_shape,
                      activation='relu', padding='same', use_bias=False)(up)
        # conv = Conv2D(n_layers, filter_shape,
                      # activation='relu', padding='same', use_bias=False)(conv)
        conv = BatchNormalization()(conv)

    conv = Conv2D(n_class, (1, 1), activation='softmax')(conv)

    conv = Lambda(lambda x: x[:, 3:-3, 3:-3])(conv)
    model = Model(inputs=[inputs], outputs=[conv])
    model.compile(optimizer=Adam(lr=1e-5),
                  loss="categorical_crossentropy", metrics=['accuracy'])

    return model


print("Creating model")
model = get_u_net((256, 256), 3, 2)
print("-" * 30)

print("Plotting model")
plot_model(model, to_file='model.png')
print("-" * 30)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=0),
    # ModelCheckpoint('model.h5',
                    # monitor='val_loss',
                    # save_best_only=True,
                    # verbose=0),
    PlotLosses('loss.png'),
]

print('Fitting model...')
history = model.fit(imgs_train,
                    imgs_mask_train,
                    # validation_split=0.2,
                    callbacks=callbacks,
                    validation_data=(imgs_validation, imgs_mask_validation),
                    batch_size=4,
                    epochs=20,
                    verbose=1,
                    shuffle=True)
print('-' * 30)

print("Saving train results to csv...")
pd.DataFrame(history.history).to_csv("train.csv", sep=",")
print('-' * 30)

print("Saving train results to png...")
model_history_to_png(history, 'acc_loss.png')
print('-' * 30)


print('Saving example predictions...')
preds = model.predict(imgs_validation[:2], verbose=1).astype('float32')
abs_preds = (preds == preds.max(axis=3)[:,:,:,None]).astype(float)
org_img = imgs_validation_org[:2]
org_mask = imgs_mask_validation_org[:2]

fig=plt.figure(figsize=(16,12))
fig.add_subplot(1,4,1)
plt.imshow(org_img[0])
fig.add_subplot(1,4,2)
plt.imshow(org_mask[0])
fig.add_subplot(1,4,3)
plt.imshow(abs_preds[0])
fig.add_subplot(1,4,4)
plt.imshow(preds[0])

fig.add_subplot(2,4,1)
plt.imshow(org_img[1])
fig.add_subplot(2,4,2)
plt.imshow(org_mask[1])
fig.add_subplot(2,4,3)
plt.imshow(abs_preds[1])
fig.add_subplot(2,4,4)
plt.imshow(preds[1])
plt.savefig('predictions.png')
print('-' * 30)
