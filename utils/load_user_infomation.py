import os
import torch
import cv2 as cv
import numpy as np

def load_embedding_vector(user_id_path, type = 'numpy', mask = False, testing = False):
    
    assert type in ["numpy", 'torch']
    
    if testing is False:
        if mask:
            embedding_vector_path = os.path.join(user_id_path, "eye_vector.txt")
        else:
            embedding_vector_path = os.path.join(user_id_path, "face_vector.txt")
    else:
        if mask:
            embedding_vector_path = os.path.join(user_id_path, "test_eye_vector.txt")
        else:
            embedding_vector_path = os.path.join(user_id_path, "test_face_vector.txt")
        
    embedding_vector = np.loadtxt(embedding_vector_path, dtype = np.float32, delimiter = ' ')
    
    if type == 'numpy':
        return embedding_vector
    else:
        return torch.from_numpy(embedding_vector)
    
def load_user_name(user_id_path):
    name_path = os.path.join(user_id_path, 'name.txt')
    file =  open(name_path, 'r', encoding= 'utf-8')
    
    name = file.readline()
    
    return name

def load_database(database_path, mask = True):
    face_embedding_list = []
    eye_embedding_list = []
    name_list = []
    user_id_list = os.listdir(database_path)
    for user_id in user_id_list:
        user_id_path = os.path.join(database_path, user_id)
        
        face_embedding_vector = load_embedding_vector(user_id_path)
        eye_embedding_vector = load_embedding_vector(user_id_path, mask = mask)
        name = load_user_name(user_id_path)
        
        face_embedding_list.append(face_embedding_vector)
        eye_embedding_list.append(eye_embedding_vector)
        name_list.append(name)
    
    face_embedding_list = np.array(face_embedding_list)
    eye_embedding_list = np.array(eye_embedding_list)
    
    if mask:
        return user_id_list, name_list, face_embedding_list, eye_embedding_list
    return user_id_list, name_list, face_embedding_list, None
    
def load_test_data(database_path, mask = True):
    face_embedding_list = []
    eye_embedding_list = []

    user_id_list = os.listdir(database_path)
    for user_id in user_id_list:
        user_id_path = os.path.join(database_path, user_id)
        
        face_embedding_vector = load_embedding_vector(user_id_path, testing = True)
        eye_embedding_vector = load_embedding_vector(user_id_path, mask = mask, testing = True)
        
        face_embedding_list.append(face_embedding_vector)
        eye_embedding_list.append(eye_embedding_vector)
    
    face_embedding_list = np.array(face_embedding_list)
    eye_embedding_list = np.array(eye_embedding_list)
    
    if mask:
        return user_id_list, face_embedding_list, eye_embedding_list
    return user_id_list, face_embedding_list, None
    
if __name__ == '__main__':
    database_path = 'folder_test'
    user_id_list, name_list, embedding_list, eye_embedding_list = load_database(database_path)
    
    embedding_vector = torch.rand(1,512)
    
    dis = np.linalg.norm(embedding_vector.detach().cpu().numpy() - embedding_list, axis = 1)
    print(dis)
    argmin_dis = np.argmin(dis)
    
    print(user_id_list[argmin_dis], name_list[argmin_dis])