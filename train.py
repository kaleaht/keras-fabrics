import argparse
import os
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(
    description='Train Convolutional Neural Fabric.')
parser.add_argument('model_name', metavar='model_name', type=str,
                    help='name for the model')
parser.add_argument('layers', metavar='number of layers', type=int,
                    help='number of fabric layers')
parser.add_argument('filters', metavar='number of filters', type=int,
                    help='number of fabric filters')
parser.add_argument('tr_conv_kernel', metavar='convolutional kernel size',
                    type=int, help='convolutional square kernel size k')
parser.add_argument('-e', '--epochs', metavar='number of epochs', type=int,
                    help='number of epochs', default=10)

args = parser.parse_args()
model_name = args.model_name
num_layers = args.layers
num_filters = args.filters
tr_conv_kernel = args.tr_conv_kernel
num_epochs = args.epochs

from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from fabrics import Fabric
from node import Node, NodeResize, NodeResizeConv, NodeBi
from helpers import plot_model_eps, PlotLosses, plot_predictions
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

print('Creating folder for train instance')

start_dir = os.getcwd()
model_dir = start_dir + '/models/' + model_name + '/'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

print('-' * 30)

print('Loading and preprocessing training and validation data...')

path = 'local_data/'

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
os.chdir(model_dir)
fabric = Fabric(NodeBi, (256, 256, 3), (num_layers, 9), num_filters,
                tr_conv_kernel)
fabric.model.compile(optimizer=Adam(lr=0.01),
                     loss="categorical_crossentropy",
                     metrics=['accuracy'])
print("-" * 30)

print("Plotting model")
plot_model(fabric.model, to_file='model.png')
print("-" * 30)

print('Fitting model...')
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=0),
    ModelCheckpoint('model.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=0),
    PlotLosses('loss.eps'),
]

history = fabric.model.fit(imgs_train,
                           imgs_mask_train,
                           validation_data=(
                               imgs_validation, imgs_mask_validation),
                           batch_size=32,
                           epochs=num_epochs,
                           callbacks=callbacks,
                           verbose=1,
                           shuffle=True)
print('-' * 30)

print("Saving train results to csv...")
pd.DataFrame(history.history).to_csv("train.csv", sep=",")
print('-' * 30)

print("Saving train results to eps...")
plot_model_eps(history, 'acc_loss.eps')
print('-' * 30)


print('Saving example predictions...')
preds = fabric.model.predict(imgs_validation[:2], verbose=1).astype('float32')
plot_predictions(preds, 
                 imgs_validation_org,
                 imgs_mask_validation_org,
                 'predictions.png')
print('-' * 30)
os.chdir(start_dir)
