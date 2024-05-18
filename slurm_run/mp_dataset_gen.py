#! /usr/bin/python3

import os
import time
import random
import uuid

import numpy as np
import cv2
from scipy import ndimage as ndi

N_CPU = 56
N_DATA = 10
LABEL_ID_METADATA = {i: {"filename": f"r{i+1}.bmp", "name": f"r{i+1}"} for i in range(0, 488)}
dataset_out_path = "dataset_out"

def distort_image(img):
    """
    Apply distortion to image using forward mapping.
    """
    # Simulate deformation field
    N = 500
    sh = (N, N)
    img = cv2.resize(img, sh, interpolation=cv2.INTER_LINEAR)
    t = np.random.normal(size=sh)
    dx = ndi.gaussian_filter(t, 80, order=(0,1))
    dy = ndi.gaussian_filter(t, 80, order=(1,0))
    dx *= 30/dx.max()
    dy *= 30/dy.max()

    # Apply forward mapping
    yy, xx = np.indices(sh)
    xmap = (xx-dx).astype(np.float32)
    ymap = (yy-dy).astype(np.float32)
    warped = cv2.remap(img, xmap, ymap ,cv2.INTER_LINEAR)
    return warped

def random_flip(img):
    """
    Randomly flip image.
    """
    if np.random.rand() > 0.5:
        img = np.fliplr(img)
    if np.random.rand() > 0.5:
        img = np.flipud(img)
    return img

def random_rotate(img):
    """
    Randomly rotate image.
    """
    prob_dist = np.cos(np.arange(0, 360)/360*12.56)**200
    prob_dist = prob_dist / np.sum(prob_dist)
    angle = np.random.choice(np.arange(0, 360), p=prob_dist)
    img = ndi.rotate(img, angle, reshape=True)
    return img

def random_resize(img):
    """
    Randomly resize image.
    """
    scalex = np.random.uniform(0.3, 0.6)
    scaley = np.random.uniform(0.3, 0.6)
    img = cv2.resize(img, None, fx=scalex, fy=scaley, interpolation=cv2.INTER_LINEAR)
    return img


def crop_to_bbox(img: np.ndarray) -> np.ndarray:
    th = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(th)
    x, y, w, h = cv2.boundingRect(coords)
    return img[y: y+h, x: x+w]


def load_image(path: "str | int", add_noise: bool = False) -> np.ndarray:
    if isinstance(path, int):
        path = f'00_Oracle/LOBI_Roots/r{path}.bmp'
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = 255 - img  # Invert image
    if add_noise:
        img = distort_image(img)
        img = random_flip(img)
        img = random_rotate(img)
        img = random_resize(img)
        img = crop_to_bbox(img)
    return img

def merge_images(images: "list[np.ndarray]") -> np.ndarray:
    # put images into a blank canvas, which is at random position
    # we also have to output the bounding box of each image, format: (x_center, y_center, w, h)
    IMG_SIZE = 500
    n = len(images)
    canvas = np.zeros((IMG_SIZE, IMG_SIZE))
    bounding_boxes = []
    for i in range(n):
        img = images[i]
        x = np.random.randint(0, IMG_SIZE - img.shape[1])
        y = np.random.randint(0, IMG_SIZE - img.shape[0])
        canvas[y: y+img.shape[0], x: x+img.shape[1]] = np.maximum(img, canvas[y: y+img.shape[0], x: x+img.shape[1]])
        bounding_boxes.append((x + img.shape[1] // 2, y + img.shape[0] // 2, img.shape[1], img.shape[0]))
    return canvas, bounding_boxes

def get_trainable_image(path: str, n_roots: int = None) -> np.ndarray:
    # path is a folder containing images
    randomly_selected_images = [random.choice(range(len(LABEL_ID_METADATA))) for _ in range(n_roots or random.randint(1, 5))]
    images = [load_image(os.path.join(path, LABEL_ID_METADATA[i]["filename"]), add_noise=True) for i in randomly_selected_images]
    out_img, bounding_boxes = merge_images(images)

    id_ = uuid.uuid4().hex
    cv2.imwrite(os.path.join(dataset_out_path, 'images', f"{id_}.png"), out_img)
    with open(os.path.join(dataset_out_path, 'labels', f"{id_}.txt"), 'w') as f:
        for class_id, bb in zip(randomly_selected_images, bounding_boxes):
            f.write(f"{class_id} {bb[0]/500} {bb[1]/500} {bb[2]/500} {bb[3]/500}\n")


def get_batch_image(path: str, batch_size=16, multiprocessing=True) -> np.ndarray:
    if multiprocessing:
        import multiprocessing as mp
        from multiprocessing import Pool
        with Pool(mp.cpu_count()) as p:
            p.map(get_trainable_image, [path]*batch_size)

def test():
    os.makedirs(dataset_out_path, exist_ok=True)
    os.makedirs(os.path.join(dataset_out_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_out_path, 'labels'), exist_ok=True)

    path = "00_Oracle/LOBI_Roots"
    with open('label_metadata.txt', 'w') as f:
        f.write(str(LABEL_ID_METADATA))
    
    get_batch_image(path, N_DATA, multiprocessing=True)

if __name__ == '__main__':
    test()

