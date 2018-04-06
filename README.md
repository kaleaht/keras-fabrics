## Download script for the Parts Labels dataset (download.py)
Downloads the images and the labels first. After this creates npy image and
label files for train, validation and test sets. Finally the original files are
removed.

## Results for validation set 

### Pixel accuracy
| Layers / Channels | 4 (30 epochs) | 16 (100 epochs) | 64 (100 epochs) |
| :---------------: |:-------------:|:---------------:|:---------------:|
| 2					| 92            | 95              | 95              |
| 4					| 93            | 95              | 95              |
| 8					| 95            | 96              | NA              |
| 16				| 95            | 96              | NA              |


