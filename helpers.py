import sys
import os
import urllib.request
import tarfile
import shutil

import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt
import numpy as np
import keras

def download_images(dest):
    download_dir = dest + 'temp/'

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    image_urls = [
        "http://vis-www.cs.umass.edu/lfw/part_labels/parts_lfw_funneled_gt_images.tgz",
        "http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz"
    ]

    for url in image_urls:
        filename = url.split('/')[-1]
        file_path = os.path.join(download_dir, filename)

        if not os.path.exists(file_path):
            # Check if the download directory exists, otherwise create it.
            if not os.path.exists(download_dir):
                os.makedirs(download_dir)

            print("Downloading from url: " + url)
            file_path, _ = urllib.request.urlretrieve(url=url,
                                                      filename=file_path)
            print("Download finished. Extracting files.")

            tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)

            print("Done.")

        else:
            print("Data has apparently already been downloaded and unpacked.")

        print("Removing file: " + file_path)
        os.remove(file_path)

    print("Done downloading and extracting images.")

    return download_dir


def move_images(dest, download_dir):

    print("Downloading txt files")

    txt_files = [
        "http://vis-www.cs.umass.edu/lfw/part_labels/parts_train.txt",
        # "http://vis-www.cs.umass.edu/lfw/part_labels/parts_test.txt",
        "http://vis-www.cs.umass.edu/lfw/part_labels/parts_validation.txt"
    ]

    base_folder = download_dir + "lfw_funneled/"
    base_folder_labels = download_dir + "parts_lfw_funneled_gt_images/"

    for url in txt_files:
        filename = url.split('/')[-1]
        file_path = os.path.join(download_dir, filename)
        dir_name = filename.split('.')[0]

        labels_dir = dest + dir_name + '_labels/'

        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)

        x_dir = dest + dir_name + '/'

        if not os.path.exists(x_dir):
            os.makedirs(x_dir)

        if not os.path.exists(file_path):

            print("Downloading from url: " + url)
            file_path, _ = urllib.request.urlretrieve(url=url,
                                                      filename=file_path)
            print("Done.")
        else:
            print("txt file already downloaded")

        txt_file = np.loadtxt(file_path, delimiter=" ", dtype=np.str)

        images = []
        labels = []

        print("Moving files: " + url)
        for i in txt_file:

            # Some file names are not complete.
            # Let's fix that
            name_len = len(i[1])
            if name_len < 4:
                i[1] = '0' * (4 - name_len) + i[1]

            folder = i[0]
            file = i[0] + "_" + i[1] + ".jpg"
            label_file = i[0] + "_" + i[1] + ".ppm"

            os.rename(base_folder + folder + "/" + file, x_dir + file)
            os.rename(base_folder_labels + "/" +
                      label_file, labels_dir + label_file)

        print("File move done!")

def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()

def model_history_to_png(model_history, file_path):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.savefig(file_path)

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
        plt.savefig(self.file_path);
        plt.gcf().clear()
