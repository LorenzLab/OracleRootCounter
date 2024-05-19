#!/usr/bin/env python

__author__ = "Adeel Ahmad"
__email__ = "adeelahmad14@hotmail.com"
__status__ = "Production"

from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import cv2
from build_reference_table import *
from match_table import *
from find_maxima import *
import numpy as np
import os


REF_IMG_DIR = 'ref_images/'
IMG_DIR = 'images/'

class Image:
    def __init__(self, path, label=None):
        self.img = 255-imread(path).astype(np.uint8)
        # add canny
        kernel_size = 3
        self.img = cv2.GaussianBlur(self.img,(kernel_size, kernel_size), 0)
        self.img = cv2.Canny(self.img, 1, 10)
        self.label = label or path


def find_best_match(img, ref_img):
    table = buildRefTable(ref_img)
    acc = matchTable(img, table)
    val, ridx, cidx = findMaxima(acc)
    return val, ridx, cidx, acc

def load_images(img_dir):
    images = []
    for img in os.listdir(img_dir):
        images.append(Image(img_dir + img))
    return images

def main():
    ref_images = load_images(REF_IMG_DIR)
    images = load_images(IMG_DIR)

    for img_obj in images:
        detections = []
        for ref_img_obj in ref_images:
            img = img_obj.img
            ref_img = ref_img_obj.img
            val, ridx, cidx, acc = find_best_match(img, ref_img)
            if acc.max() < 65:
                print("No match found!")
                continue
            # # code for drawing bounding-box in accumulator array...

            # acc[ridx - 5:ridx + 5, cidx - 5] = val
            # acc[ridx - 5:ridx + 5, cidx + 5] = val

            # acc[ridx - 5, cidx - 5:cidx + 5] = val
            # acc[ridx + 5, cidx - 5:cidx + 5] = val

            # plt.figure(1)
            # imshow(acc)
            # plt.show()

            # code for drawing bounding-box in original image at the found location...

            # find the half-width and height of template
            hheight = np.floor(ref_img.shape[0] / 2) + 1
            hwidth = np.floor(ref_img.shape[1] / 2) + 1

            # find coordinates of the box
            rstart = int(max(ridx - hheight, 1))
            rend = int(min(ridx + hheight, img.shape[0] - 1))
            cstart = int(max(cidx - hwidth, 1))
            cend = int(min(cidx + hwidth, img.shape[1] - 1))

            # # draw the box
            # img[rstart:rend, cstart] = 255
            # img[rstart:rend, cend] = 255

            # img[rstart, cstart:cend] = 255
            # img[rend, cstart:cend] = 255
            
            # add the detection to the list in the format (label, cx, cy, w, h)
            detections.append((ref_img_obj.label, cidx, ridx, ref_img.shape[1], ref_img.shape[0]))

            # show the image
            # plt.figure(3), imshow(img)
            # plt.show()
        print("Detections for image: ", detections)


if __name__ == '__main__':
    main()
