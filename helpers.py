"""
Helper functions.
"""
import numpy as np
import keras
import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt


def plot_model(model):
    hist = model.history
    _, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize hist for accuracy
    axs[0].plot(range(1, len(hist['acc']) + 1), hist['acc'])
    axs[0].plot(range(1, len(hist['val_acc']) + 1), hist['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(hist['acc']) + 1),
                      len(hist['acc']) / 10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize hist for loss
    axs[1].plot(range(1, len(hist['loss']) + 1), hist['loss'])
    axs[1].plot(range(1, len(hist['val_loss']) + 1), hist['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(hist['loss']) + 1),
                      len(hist['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()


def plot_model_eps(model, file_path):
    hist = model.history
    acc = hist['acc']
    val_acc = hist['val_acc']
    loss = hist['loss']
    val_loss = hist['val_loss']
    x_range = range(1, len(acc) + 1)


    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(x_range, acc)
    axs[0].plot(x_range, val_acc)
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(acc) + 1), len(acc) / 10)
    axs[0].legend(['train', 'val'], loc='best')
    
    # summarize history for loss
    axs[1].plot(x_range, loss)
    axs[1].plot(x_range, val_loss)
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(loss) + 1), len(loss) / 10)
    axs[1].legend(['train', 'val'], loc='best')

    plt.savefig(file_path, format='eps')

def plot_predictions(preds, imgs_validation_org, imgs_mask_validation_org, file_path):
    abs_preds = (preds == preds.max(axis=3)[:, :, :, None]).astype(float)
    org_img = imgs_validation_org[:2]
    org_mask = imgs_mask_validation_org[:2]

    fig = plt.figure(figsize=(16, 12))
    fig.add_subplot(1, 4, 1)
    plt.imshow(org_img[0])
    fig.add_subplot(1, 4, 2)
    plt.imshow(org_mask[0])
    fig.add_subplot(1, 4, 3)
    plt.imshow(abs_preds[0])
    fig.add_subplot(1, 4, 4)
    plt.imshow(preds[0])

    fig.add_subplot(2, 4, 1)
    plt.imshow(org_img[1])
    fig.add_subplot(2, 4, 2)
    plt.imshow(org_mask[1])
    fig.add_subplot(2, 4, 3)
    plt.imshow(abs_preds[1])
    fig.add_subplot(2, 4, 4)
    plt.imshow(preds[1])
    plt.savefig('predictions.png')

class PlotLosses(keras.callbacks.Callback):
    def __init__(self, file_path):
        self.validation_data = None
        self.model = None
        self.file_path = file_path

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.savefig(self.file_path, format='eps')
        plt.gcf().clear()
