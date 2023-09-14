# Copyright (c) Meta Platforms, Inc. and affiliates.

import os

import cv2
import numpy as np
from PIL import Image

ckbd_cache = {}


def checkerboard(H, W):
    key = f"{H}x{W}"
    if key in ckbd_cache:
        return ckbd_cache[key]

    board = np.ones([H, W, 3]) * 0.8
    sz = H // 16
    for y in range(0, H, sz):
        for x in range(0, W, sz):
            if (x//sz+y//sz) % 2 == 0:
                continue
            board[y:y+sz, x:x+sz] = 0.6
    ckbd_cache[key] = board
    return board


def read_image(fp, scale: float = 1, resample: int = Image.BILINEAR) -> Image.Image:
    img = Image.open(fp)
    if scale == 1:
        return img

    w = int(np.round(img.width * scale))
    h = int(np.round(img.height * scale))
    return img.resize((w, h), resample=resample)


def read_image_np(fp, scale: float = 1, resample: int = Image.BILINEAR) -> np.ndarray:
    img = read_image(fp, scale, resample)
    return np.array(img, dtype=np.float32) / 255


def save_image(path, image: Image.Image):
    ext = os.path.splitext(path)[1]
    if ext == ".jpg":
        params = {"quality": 95}
    elif ext == ".png":
        params = {"optimize": True}
    elif ext == ".webp":
        params = {"quality": 95, "method": 6}

    image.save(path, **params)


def save_image_np(path, data: np.ndarray):
    data = np.clip(data * 255, 0, 255).astype(np.uint8)
    image = Image.fromarray(data)
    save_image(path, image)


def normalize_array(array: np.ndarray, pmin=0, pmax=100, gamma=1) -> np.ndarray:
    dmin, dmax = np.percentile(array, [pmin, pmax])
    array = np.clip((array - dmin) / (dmax - dmin), 0, 1)
    if gamma != 1:
        array = np.power(array, gamma)
    return array


def visualize_array(array, color_map=cv2.COLORMAP_JET):
    """
    array: [H, W], values in range [0, 1]
    color_map: cv2.COLORMAP_*
    """
    # NOTE: casting float array with NaNs to uint8 results in 0s
    scaled = (array**0.5 * 255).astype(np.uint8)
    colored = (cv2.applyColorMap(scaled, color_map) / 255) ** 2.2 * 255
    converted = cv2.cvtColor(colored.astype(np.float32), cv2.COLOR_BGR2RGB) / 255

    return converted
