import torch
import numpy as np
from typing import Tuple, Optional, Union

def load_weights(model: torch.nn.Module, weights, device = 'cpu', eval = True):
    
    model.load_state_dict(
        torch.load(weights, map_location = 'cpu')
    )
    
    if eval:
        model.eval()
    
    model.to(device)
    
    return model

def calculate_angle(v1: Union[list, tuple, torch.Tensor, np.ndarray],
                    v2: Union[list, tuple, torch.Tensor, np.ndarray]):
    '''
    Calculate the angle between 2 vectors v1 and v2
    '''
    if isinstance(v1, torch.Tensor):
        v1 = v1.numpy()
    else: 
        v1 = np.array(v1)
    if isinstance(v2, torch.Tensor):
        v2 = v2.numpy()
    else: 
        v2 = np.array(v2)
        
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    rad = np.arccos(cosine)

    degrees = np.degrees(rad)
    
    return np.round(degrees)