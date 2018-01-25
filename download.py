import sys
import os
import urllib.request
import tarfile
import shutil

import numpy as np
import skimage
import skimage.io
from skimage.transform import resize


def download(dest):
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

    print("Downloading txt files")

    txt_files = [
        "http://vis-www.cs.umass.edu/lfw/part_labels/parts_train.txt",
        "http://vis-www.cs.umass.edu/lfw/part_labels/parts_test.txt",
        "http://vis-www.cs.umass.edu/lfw/part_labels/parts_validation.txt"
    ]

    base_folder = download_dir + "lfw_funneled/"
    base_folder_labels = download_dir + "parts_lfw_funneled_gt_images/"

    for url in txt_files:
        filename = url.split('/')[-1]
        file_path = os.path.join(download_dir, filename)

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

            img = skimage.io.imread(base_folder + folder + "/" + file)
            img = resize(img, (256, 256))
            images.append(img)

            label = skimage.io.imread(base_folder_labels + "/" + label_file)
            label = resize(label, (256, 256))
            labels.append(label)

        images = np.array(images)
        labels = np.array(labels)

        save_loc = dest + file_path.split('/')[-1][:-4]
        np.save(save_loc, images)
        print("Saved to: " + save_loc)

        save_loc_labels = dest + file_path.split('/')[-1][:-4] + '_labels'
        np.save(save_loc_labels, np.round(labels))
        print("Saved to: " + save_loc_labels)

    print("Removing temp folder: " + download_dir)
    shutil.rmtree(download_dir)
    print("Done.")


folder = 'data/'
download(folder)
