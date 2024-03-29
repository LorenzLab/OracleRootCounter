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
    img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_LINEAR)
    return img

def random_resize(img):
    """
    Randomly resize image.
    """
    scalex = np.random.uniform(0.5, 2)
    scaley = np.random.uniform(0.5, 2)
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
    return img

def test():
    test_distort()

if __name__ == '__main__':
    image = load_image(1, add_noise=True)
    plt.imshow(image, cmap='gray')
    plt.show()
