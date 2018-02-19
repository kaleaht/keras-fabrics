from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

sys.path.append('../..')
from fabrics import Fabric
from helpers import model_history_to_png, PlotLosses

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

print("Creating model")
fabric = Fabric((256, 256, 3), (2, 9), 4, kernel_shape=(3,3), channels_double=False)
fabric.model.compile(optimizer=Adam(),
                     loss="categorical_crossentropy",
                     metrics=['accuracy'])
print("-" * 30)

print("Plotting model")
plot_model(fabric.model, to_file='model.png')
print("-" * 30)

print('Fitting model...')
callbacks = [
    # EarlyStopping(monitor='val_loss', patience=10, verbose=0),
    ModelCheckpoint('model.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=0),
    PlotLosses('loss.png'),
]

history = fabric.model.fit(imgs_train,
                           imgs_mask_train,
                           validation_data=(
                               imgs_validation, imgs_mask_validation),
                           batch_size=32,
                           epochs=500,
                           callbacks=callbacks,
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
preds = fabric.model.predict(imgs_validation[:2], verbose=1).astype('float32')
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
