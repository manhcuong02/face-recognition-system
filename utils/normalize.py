import numpy as np
import torch
import cv2 as cv
from typing import Union

def invert_normalize(images: torch.Tensor):
    img_shape = images.shape
    if len(images.shape) == 3:
        images = images[None, ...]
    
    images = images.permute(0, 2, 3, 1).detach().cpu().numpy()
    
    images = images * 128 + 127.5

    images = images.astype(np.uint8)
    
    if len(img_shape) == 3:
        return images[0]

    return images

def normalize_rgb_image_facenet(img: np.ndarray, size = (160, 160)) -> torch.Tensor:
    "convert from PIL Image to tensor"
    img = cv.resize(img, size)
    img = img.astype(np.float32)
    img = (img - 127.5)/128.
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)
    return img_tensor

def normalize_rgb_image_arcface(img: np.ndarray, size = (112, 112)) -> torch.Tensor:
    "convert from PIL Image to tensor"
    img = cv.resize(img, size)
    img = img.astype(np.float32)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)
    img_tensor.div_(255).sub_(0.5).div_(0.5)
    return img_tensor

def normalize_batch_rgb_images_facenet(img: np.ndarray) -> torch.Tensor:
    "convert from PIL Image to tensor"
    img = img.astype(np.float32)
    img = (img - 127.5)/128.
    img_tensor = torch.from_numpy(img).permute(0, 3, 1, 2)
    return img_tensor

def normalize_batch_rgb_images_arcface(img: np.ndarray, size = (112, 112)) -> torch.Tensor:
    "convert from PIL Image to tensor"
    img = cv.resize(img, size)
    img = img.astype(np.float32)
    img_tensor = torch.from_numpy(img).permute(0, 3, 1, 2)
    img_tensor.div_(255).sub_(0.5).div_(0.5)
    return img_tensor

def convert_mtcnn2arcface_norm(img: Union[np.ndarray, torch.Tensor]):
    return img * 1.004