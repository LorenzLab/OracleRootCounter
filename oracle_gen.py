import os
import time
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

def test_distort():
    test_img = np.zeros((500, 500))
    test_img[::10, :] = 1
    test_img[:, ::10] = 1
    test_img = ndi.gaussian_filter(test_img, 0.5)
    warped = distort_image(test_img)

    plt.imshow(warped, cmap='gray')
    plt.show()

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
        x = np.random.randint(0, IMG_SIZE - img.shape[0])
        y = np.random.randint(0, IMG_SIZE - img.shape[1])
        canvas[x: x+img.shape[0], y: y+img.shape[1]] = np.maximum(img, canvas[x: x+img.shape[0], y: y+img.shape[1]])
        bounding_boxes.append((x+img.shape[0]//2, y+img.shape[1]//2, img.shape[0], img.shape[1]))
    return canvas, bounding_boxes

def get_trainable_image(path: str, n_roots: int = None) -> np.ndarray:
    # path is a folder containing images
    images = os.listdir(path)
    randomly_selected_images = [random.choice(range(len(images))) for _ in range(n_roots or random.randint(1, 5))]
    images = [load_image(os.path.join(path, images[i]), add_noise=True) for i in randomly_selected_images]
    out_img, bounding_boxes = merge_images(images)
    return out_img, bounding_boxes, randomly_selected_images

def get_batch_image(path: str, batch_size=16, multiprocessing=True) -> np.ndarray:
    if multiprocessing:
        import multiprocessing as mp
        from multiprocessing import Pool
        with Pool(mp.cpu_count()) as p:
            data = p.map(get_trainable_image, [path]*batch_size)
        images, bounding_boxes, labels = zip(*data)
    else:        
        # path is a folder containing images
        images = []
        bounding_boxes = []
        labels = []
        for _ in range(batch_size):
            out_img, bb, l = get_trainable_image(path)
            images.append(out_img)
            bounding_boxes.append(bb)
            labels.append(l)
    return images, bounding_boxes, labels

def test():
    path = "00_Oracle/LOBI_Roots"
    with open('file_list_order.txt', 'w') as f:
        for line in os.listdir(path):
            f.write(f"{line}\n")
    fig = plt.figure(figsize=(13, 13))
    
    ss = time.time()
    out_img, bounding_boxes, randomly_selected_images = get_batch_image(path, multiprocessing=True)
    print(f"Time taken: {time.time() - ss:.2f}s")
    
    for en, (i, j, k) in enumerate(zip(out_img, bounding_boxes, randomly_selected_images), 1):
        plt.subplot(4, 4, en)
        plt.axis('off')
        plt.imshow(i, cmap='gray')
        plt.title(f"Labels: {k}")
    plt.show()

if __name__ == '__main__':
    test()
