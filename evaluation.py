import torch
from facenet import *
from utils.load_user_infomation import *
from typing import Union
import numpy as np
import argparse
import matplotlib.pyplot as plt
from utils.plot import *

def norm_l2(embedding_list: np.ndarray, test_embedding_list: np.ndarray):
    test_embedding_list = test_embedding_list.reshape(test_embedding_list.shape[0], test_embedding_list.shape[1], 1, test_embedding_list.shape[2])

    norm = np.linalg.norm(test_embedding_list - embedding_list, axis = -1)
        
    argmin_idx = np.argmin(norm, axis = -1)
    
    min_dist = np.min(norm, axis = -1)
    
    return argmin_idx, min_dist

def cosine_similarity(x, y):
    # Tính norm của từng vector trong x và y
    y_shape = y.shape
    
    y = y.reshape(-1, 512)
    
    norm_x = np.linalg.norm(x, axis=1, keepdims=True)
    norm_y = np.linalg.norm(y, axis=1, keepdims=True)

    # Chia mỗi vector cho độ dài của nó
    normalized_x = x / norm_x
    normalized_y = y / norm_y

    # Tính cosine similarity bằng cách nhân tich vô hướng các vector
    similarity = np.dot(normalized_x, normalized_y.T)
        
    similarity = similarity.T
        
    argmax_dist = np.argmax(similarity, axis = -1)
    
    max_dist = np.max(similarity, axis = -1)
    
    return argmax_dist.reshape(*y_shape[:2]), max_dist.reshape(*y_shape[:2])

def main(opt):
    user_id_list, name_list, face_embedding_list, eye_embedding_list = load_database(opt.database_path, opt.mask)
    test_user_id_list, test_face_embedding_list, test_eye_embedding_list = load_test_data(opt.testdata_path, opt.mask)

    user_id_list = np.array(user_id_list)
    test_user_id_list = np.array(user_id_list)

    print(face_embedding_list.shape, test_face_embedding_list.shape)
    
    argmin_face, min_dist_face = norm_l2(face_embedding_list, test_face_embedding_list)
    argmin_eye, min_dist_eye = norm_l2(eye_embedding_list, test_eye_embedding_list)
    argmin_both_idx = np.where(min_dist_face < min_dist_eye, argmin_face, argmin_eye)
    
    predict_user_face_id = user_id_list[argmin_face]
    predict_eye_use_id = user_id_list[argmin_eye]
    predict_both_user_id = user_id_list[argmin_both_idx]

    test_user_id_list = np.tile(test_user_id_list, (predict_user_face_id.shape[1], 1)).T

    face_accuracy = np.sum(test_user_id_list == predict_user_face_id, dtype = np.float32)/np.prod(test_user_id_list.shape)
    eye_accuracy = np.sum(test_user_id_list == predict_eye_use_id, dtype=np.float32)/np.prod(test_user_id_list.shape)
    both_accuracy = np.sum(test_user_id_list == predict_both_user_id, dtype=np.float32)/np.prod(test_user_id_list.shape)
    
    print("face accuracy: {}, eye_accuracy: {}, both_accuracy: {}".format(face_accuracy, eye_accuracy, both_accuracy))

    plot_confusion_matrix(test_user_id_list.reshape(-1), predict_user_face_id.reshape(-1), label = test_user_id_list[:,0], title = "Face accuracy")
    plot_confusion_matrix(test_user_id_list.reshape(-1), predict_eye_use_id.reshape(-1), label = test_user_id_list[:,0], title = "Eye accuracy")
    plot_confusion_matrix(test_user_id_list.reshape(-1), predict_both_user_id.reshape(-1), label = test_user_id_list[:,0], title = "Combine Both accuracy")
    plt.show()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", type=str, default = "quant_database", help = "Database path")
    parser.add_argument("--testdata_path", type=str, default = "quant_database", help = "Test data path")
    parser.add_argument("--mask", action = 'store_true', help = "Process mask for face")

    opt = parser.parse_args()
        
    main(opt)