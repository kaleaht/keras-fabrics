from keras.layers import Input, Add, Conv2D
import numpy as np
from keras.models import Model
from keras.optimizers import Adam
from fabrics import UpSample, DownSample, SameRes, Fabric, Node
from keras import backend as K
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

print('-' * 30)
print('Loading and preprocessing training and validation data...')
print('-' * 30)

path = 'local_data/'

imgs_train = np.load(path + 'parts_train.npy')
imgs_mask_train = np.load(path + 'parts_train_labels.npy')

imgs_train = imgs_train.astype('float32')
mean = np.mean(imgs_train)  # mean for data centering
std = np.std(imgs_train)  # std for data normalization

imgs_train -= mean
imgs_train /= std

imgs_mask_train = imgs_mask_train.astype('float32')

imgs_validation = np.load(path + 'parts_validation.npy')
imgs_mask_validation = np.load(path + 'parts_validation_labels.npy')

imgs_validation -= mean
imgs_validation /= std

imgs_mask_validation = imgs_mask_validation.astype('float32')

flip = np.flip(imgs_train, axis=2)
flip_mask = np.flip(imgs_mask_train, axis=2)

imgs_train = np.concatenate([imgs_train, flip])
imgs_mask_train = np.concatenate([imgs_mask_train, flip_mask])


fabric = Fabric((256, 256, 3), (2, 9), 4, channels_double=False)
print('-' * 30)
print('Fitting model...')
print('-' * 30)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, verbose=0),
    ModelCheckpoint('trained_models/2x9.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=0),
]

history = fabric.model.fit(imgs_train,
                           imgs_mask_train,
                           validation_data=(
                               imgs_validation, imgs_mask_validation),
                           batch_size=32,
                           epochs=1000,
                           callbacks=callbacks,
                           verbose=1,
                           shuffle=True)
