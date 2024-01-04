import numpy as np
import torch
from typing import Union

def L2_distance(x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]):
    
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
            
    l2_dist = np.linalg.norm(x - y, axis = 1)
    
    min_dist = np.min(l2_dist)
    argmin_dist = np.argmin(l2_dist)
    
    return min_dist, argmin_dist

def cosine_similarity(x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    
    rad = (np.dot(x, y.T)/(np.linalg.norm(x, axis = 1, keepdims=True) * np.linalg.norm(y, axis = 1, keepdims=True)))    
    
    max_dist = np.max(rad)
    argmax_dist = np.argmax(rad)
    
    return max_dist, argmax_dist
    
