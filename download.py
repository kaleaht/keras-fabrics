"""
Download script for the Parts Labels dataset.
Downloads the images and the labels first. After this creates npy image and
label files for train, validation and test sets. Finally the original files are
removed.
"""
import os
import urllib.request
import tarfile
import shutil
import numpy as np
import skimage
import skimage.io
from skimage.util import pad

DEST = 'data/'
DOWNLOAD_DIR = DEST + 'temp/'

if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

BASE_URL = "http://vis-www.cs.umass.edu/lfw/"
IMG_URLS = [
    BASE_URL + "part_labels/parts_lfw_funneled_gt_images.tgz",
    BASE_URL + "/lfw-funneled.tgz"
]

for url in IMG_URLS:
    filename = url.split('/')[-1]
    file_path = os.path.join(DOWNLOAD_DIR, filename)

    if not os.path.exists(file_path):
        # Check if the download directory exists, otherwise create it.
        if not os.path.exists(DOWNLOAD_DIR):
            os.makedirs(DOWNLOAD_DIR)

        print("Downloading from url: " + url)
        file_path, _ = urllib.request.urlretrieve(url=url,
                                                  filename=file_path)
        print("Download finished. Extracting files.")

        tarfile.open(name=file_path, mode="r:gz").extractall(DOWNLOAD_DIR)

        print("Done.")

    else:
        print("Data has apparently already been downloaded and unpacked.")

    print("Removing file: " + file_path)
    os.remove(file_path)

print("Downloading the text files.")

TXT_URLS = [
    BASE_URL + "/part_labels/parts_train.txt",
    BASE_URL + "/part_labels/parts_test.txt",
    BASE_URL + "/part_labels/parts_validation.txt"
]

BASE_FOLDER = DOWNLOAD_DIR + "lfw_funneled/"
BASE_FOLDER_LABELS = DOWNLOAD_DIR + "parts_lfw_funneled_gt_images/"

for url in TXT_URLS:
    filename = url.split('/')[-1]
    file_path = os.path.join(DOWNLOAD_DIR, filename)

    if not os.path.exists(file_path):

        print("Downloading from url: " + url)
        file_path, _ = urllib.request.urlretrieve(url=url,
                                                  filename=file_path)
        print("Done.")
    else:
        print("The text file", filename, "already downloaded")

    txt_file = np.loadtxt(file_path, delimiter=" ", dtype=np.str)

    images = []
    labels = []

    print("Creating npy file from txt file: " + file_path)
    for i in txt_file:

        # Some file names are not complete.
        # Let's fix that
        name_len = len(i[1])
        if name_len < 4:
            i[1] = '0' * (4 - name_len) + i[1]

        folder = i[0]
        file = i[0] + "_" + i[1] + ".jpg"
        label_file = i[0] + "_" + i[1] + ".ppm"

        img = skimage.io.imread(BASE_FOLDER + folder + "/" + file)
        # Original image size is 250px x 250px. We pad the image with 0,
        # so we get 265x256 images.
        img = pad(img, ((3, 3), (3, 3), (0, 0)), mode='constant')
        images.append(img)

        label = skimage.io.imread(BASE_FOLDER_LABELS + "/" + label_file)
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    save_loc = DEST + file_path.split('/')[-1][:-4]
    np.save(save_loc, images)
    print("Saved to: " + save_loc)

    save_loc_labels = DEST + file_path.split('/')[-1][:-4] + '_labels'
    np.save(save_loc_labels, np.round(labels))
    print("Saved to: " + save_loc_labels)

print("Removing temp folder: " + DOWNLOAD_DIR)
shutil.rmtree(DOWNLOAD_DIR)
print("Done.")
