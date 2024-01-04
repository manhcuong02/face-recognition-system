import math
import numpy as np
from typing import Union, List
from utils import *
from utils.rotation import *
from facenet import MTCNN
from utils.normalize import *

def detect_user_face(mtcnn: MTCNN, rgb_frame, return_landmarks = False, onnx = False):
    face, box, landmark = mtcnn(rgb_frame, return_landmarks = True)
    if return_landmarks: 
        if face is None:
            return None, None, None
        return face, box[0], landmark[0]
    return face

def detect_eye_user(rgb_frame: np.ndarray, box: Union[List, np.ndarray],
                    landmark: Union[List, np.ndarray], padding_factor: float = 0.09375,
                    model_name = "facenet",
                    return_eye_image = False, return_landmark = False):
    left_eye = landmark[0]
    right_eye = landmark[1]
    
    rad = math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
    angle = math.degrees(rad)
    
    width_image, height_image = rgb_frame.shape[1], rgb_frame.shape[0]
    
    rotation_matrix = calculate_rotation_matrix(angle, width_image, height_image)
    rotated_image = rotate_image(rgb_frame, rotation_matrix)
    rotated_box = rotate_box(*box, rotation_matrix, width_image, height_image)
    rotated_landmark = rotate_landmark(landmark, rotation_matrix)
    
    rotated_right_eye = rotated_landmark[1]
            
    eye_box = [rotated_box[0], rotated_box[1], rotated_box[2], rotated_right_eye[1] + (rotated_box[3] - rotated_box[1])*padding_factor]

    eye_box = np.array(eye_box, dtype = np.int16)
    
    eye_region = rotated_image[eye_box[1]: eye_box[3], eye_box[0] : eye_box[2]]
    
    if model_name == 'facenet':
        normalized_eye_region = normalize_rgb_image_facenet(eye_region)
    elif model_name == 'arcface':
        normalized_eye_region = normalize_rgb_image_arcface(eye_region)
    # normalized_eye_region = normalize_rgb_image_facenet(eye_region)
    
    if return_eye_image:
        return normalized_eye_region, eye_region
    elif return_landmark:
        return normalized_eye_region, rotated_landmark
    else:
        return normalized_eye_region
    
def detect_orientation(landmarks: Union[List, np.ndarray], frontal_range = [20, 65]):
    '''
    returns True if the face is facing forward 
    '''
    left_eye = np.array(landmarks[0], dtype = np.float32)
    right_eye = np.array(landmarks[1], dtype = np.float32)
    nose = np.array(landmarks[2], dtype = np.float32)
    
    left2right_eye = right_eye - left_eye
    lefteye2nose = nose - left_eye
        
    left_angle = calculate_angle(left2right_eye, lefteye2nose)
    
    right2left_eye = left_eye - right_eye
    righteye2nose = nose - right_eye
        
    right_angle = calculate_angle(right2left_eye, righteye2nose)
    
    # print(f"Right angle: {right_angle: .2f}, left angle: {left_angle: .2f}")
    
    if frontal_range[0] <= left_angle <= frontal_range[1] \
            and frontal_range[0] <= right_angle <= frontal_range[1]:
        return True
    
    return False
    