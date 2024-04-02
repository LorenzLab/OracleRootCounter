import os
import random

import numpy as np
import cv2
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

def distort_image(img):
    """
    Apply distortion to image using forward mapping.
    """
    # Simulate deformation field
    N = 500
    sh = (N, N)
    img = cv2.resize(img, sh, interpolation=cv2.INTER_LINEAR)
    t = np.random.normal(size=sh)
    dx = ndi.gaussian_filter(t, 40, order=(0,1))
    dy = ndi.gaussian_filter(t, 40, order=(1,0))
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
    angle = np.random.uniform(0, 360)
    img = ndi.rotate(img, angle, reshape=False)
    return img

def random_resize(img):
    """
    Randomly resize image.
    """
    scalex = np.random.uniform(0.3, 0.6)
    scaley = np.random.uniform(0.3, 0.6)
    img = cv2.resize(img, None, fx=scalex, fy=scaley, interpolation=cv2.INTER_LINEAR)
    return img

def test_distort():
    test_img = np.zeros((500, 500))
    test_img[::10, :] = 1
    test_img[:, ::10] = 1
    test_img = ndi.gaussian_filter(test_img, 0.5)
    warped = distort_image(test_img)

    plt.imshow(warped, cmap='gray')
    plt.show()

def load_image(path: str | int, add_noise: bool = False) -> np.ndarray:
    if isinstance(path, int):
        path = f'00_Oracle/LOBI_Roots/r{path}.bmp'
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = 255 - img  # Invert image
    if add_noise:
        img = distort_image(img)
        img = random_flip(img)
        img = random_rotate(img)
        img = random_resize(img)
    return img

def merge_images(images: list[np.ndarray]) -> np.ndarray:
    # put images into a blank canvas, which is at random position
    # we also have to output the bounding box of each image, format: (x_center, y_center, w, h)
    IMG_SIZE = 500
    n = len(images)
    canvas = np.zeros((IMG_SIZE, IMG_SIZE))
    bounding_boxes = []
    for i in range(n):
        img = images[i]
        x = np.random.randint(0, IMG_SIZE - img.shape[0])
        y = np.random.randint(0, IMG_SIZE - img.shape[1])
        canvas[x:x+img.shape[0], y:y+img.shape[1]] = np.maximum(img, canvas[x:x+img.shape[0], y:y+img.shape[1]])
        bounding_boxes.append((x+img.shape[0]//2, y+img.shape[1]//2, img.shape[0], img.shape[1]))
    return canvas, bounding_boxes

def get_batch_image(path: str) -> np.ndarray:
    # path is a folder containing images
    images = os.listdir(path)
    randomly_selected_images = [random.choice(range(len(images))) for _ in range(random.randint(1, 5))]
    images = [load_image(os.path.join(path, images[i]), add_noise=True) for i in randomly_selected_images]
    out_img, bounding_boxes = merge_images(images)
    return out_img, bounding_boxes, randomly_selected_images

def test():
    test_distort()

if __name__ == '__main__':
    path = "00_Oracle/LOBI_Roots"
    out_img, bounding_boxes, randomly_selected_images = get_batch_image(path)
    for i in randomly_selected_images:
        print(os.listdir(path)[i])
    plt.imshow(out_img, cmap='gray')
    plt.show()
