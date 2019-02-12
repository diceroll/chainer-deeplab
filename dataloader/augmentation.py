import numpy as np
from scipy.misc import imresize
from scipy.ndimage import rotate


def mirror(img, label):
    return img[:, ::-1, :], label[:, ::-1, :]


def flip(img, label):
    return img[::-1, :, :], label[::-1, :, :]


def gamma_correction(img, label):
    rnd = np.random.randn()
    if rnd != 0:
        img = np.frompyfunc(lambda x, y: 255. * (x / 255.) ** y if x != 0 else x, 2, 1)(img, rnd)
        img = np.uint8(img)

    return np.float32(img), label


def random_rotate(img, label, task, angle=5):
    if task == 'depth':
        cval = 0
    elif task == 'semantic':
        cval = -1
    elif task == 'normal':
        raise NotImplementedError
    else:
        raise ValueError
    rnd = np.random.randint(-angle, angle + 1)
    if rnd != 0:
        img = rotate(img, rnd, reshape=False)
        label = rotate(label, rnd, reshape=False, cval=cval)

    return img, label


def cutout(img, label, mask_size=None):
    h, w, _ = img.shape
    if mask_size is None:
        mask_size = np.min((h, w)) // 3
    rnd1 = np.random.randint(-mask_size // 2, h - mask_size // 2)
    rnd2 = np.random.randint(-mask_size // 2, w - mask_size // 2)
    sl1 = slice(np.max((rnd1, 0)), np.min((rnd1 + mask_size, h)))
    sl2 = slice(np.max((rnd2, 0)), np.min((rnd2 + mask_size, w)))
    img[sl1, sl2] = img.mean()

    return img, label


def random_erasing(img, label, area_ratio=(0.02, 0.4), aspect_ratio=(0.3, 3)):
    h, w, _ = img.shape
    area = h * w
    aspect = np.min(aspect_ratio) + np.random.rand() * np.abs(np.diff(aspect_ratio))

    min_size = np.max((np.sqrt(area * np.min(area_ratio) / aspect), 1 // aspect + (1 % aspect > 0)))
    max_size = np.min((np.sqrt(area * np.max(area_ratio) / aspect), w // aspect))
    mask_size = np.random.randint(int(min_size), int(max_size + 1))

    rnd1 = np.random.randint(0, h - mask_size)
    rnd2 = np.random.randint(0, w - int(mask_size * aspect))
    sl1 = slice(rnd1, rnd1 + mask_size)
    sl2 = slice(rnd2, rnd2 + int(mask_size * aspect))
    img[sl1, sl2] = np.random.randint(0, 256)

    return img, label
