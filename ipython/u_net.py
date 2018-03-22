# get_ipython().run_line_magic('matplotlib', 'inline')
from __future__ import print_function
import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, Add, Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.engine.topology import Layer
import keras
from matplotlib import pyplot as plt
# from IPython.display import clear_output
from keras.utils import plot_model
from helpers import plot_model_history, PlotLosses
# from IPython.display import Image as DImage

print('-' * 30)
print('Loading and preprocessing training and validation data...')
print('-' * 30)

path = '../local_data/'

imgs_train = np.load(path + 'parts_train.npy').astype('float32')
imgs_mask_train = np.load(path + 'parts_train_labels.npy').astype('float32')

imgs_train = imgs_train.astype('float32')
mean = np.mean(imgs_train)  # mean for data centering
std = np.std(imgs_train)  # std for data normalization

imgs_train -= mean
imgs_train /= std

imgs_mask_train = imgs_mask_train.astype('float32')


imgs_validation = np.load(path + 'parts_validation.npy').astype('float32')
imgs_mask_validation = np.load(path + 'parts_validation_labels.npy').astype('float32')

imgs_validation -= mean
imgs_validation /= std

imgs_mask_validation = imgs_mask_validation.astype('float32')

# ## Data augmentation
#
# Horizontal flip

flip = np.flip(imgs_train, axis=2)
flip_mask = np.flip(imgs_mask_train, axis=2)

imgs_train = np.concatenate([imgs_train, flip])
imgs_mask_train = np.concatenate([imgs_mask_train, flip_mask])


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
        conv = Conv2D(n_layers, filter_shape, activation='relu',
                      padding='same', use_bias=False)(conv)
        conv = BatchNormalization()(conv)
        conv_layers.insert(0, conv)
        input_h = MaxPooling2D(pool_size=(2, 2))(conv)

    n_layers *= 2
    conv = Conv2D(n_layers, filter_shape, activation='relu',
                  padding='same', use_bias=False)(input_h)
    conv = Conv2D(n_layers, filter_shape, activation='relu',
                  padding='same', use_bias=False)(conv)

    for j in range(0, depth):
        n_layers = int(n_layers / 2)

        conv_trans = Conv2DTranspose(
            n_layers, kernel_shape, strides=(2, 2), padding='same')(conv)
        up = keras.layers.add([conv_trans, conv_layers[j]])
        conv = Conv2D(n_layers, filter_shape,
                      activation='relu', padding='same', use_bias=False)(up)
        conv = Conv2D(n_layers, filter_shape,
                      activation='relu', padding='same', use_bias=False)(conv)
        conv = BatchNormalization()(conv)

    conv = Conv2D(n_class, (1, 1), activation='softmax')(conv)

    conv = Lambda(lambda x: x[:, 3:-3, 3:-3])(conv)
    model = Model(inputs=[inputs], outputs=[conv])
    model.compile(optimizer=Adam(lr=1e-5),
                  loss="categorical_crossentropy", metrics=['accuracy'])

    return model


model = get_u_net((256, 256), 3, 8)

plot_model(model, to_file='../images/u-net_double_conv.png')

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=0),
    # PlotLosses('../images/u_net_loss.png'),
]

history = model.fit(imgs_train,
                    imgs_mask_train,
                    # validation_split=0.2,
                    callbacks=callbacks,
                    validation_data=(imgs_validation, imgs_mask_validation),
                    batch_size=4,
                    epochs=20,
                    verbose=1,
                    shuffle=True)
