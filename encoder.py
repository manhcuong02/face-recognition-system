
import torch
from PIL import Image 
import numpy as np
from facenet import *
from arcface import *

import os
from typing import Union
from utils.normalize import *
from utils.clustering import *
from utils import *
from tqdm import tqdm
import cv2 as cv

def encode_face_images_facenet(model: InceptionResnetV1, images: Union[np.ndarray, Image.Image], classify = False, device = 'cpu'):
    
    if isinstance(images, Image.Image):
        images = np.array(images)
            
    if len(images.shape) == 3:
        images = images[None, ...]
    
    images = normalize_batch_rgb_images_facenet(images)
    
    model.eval()
    model.classify = classify
    
    images = images.to(device)
    embedding_vectors = model(images)
    
    return embedding_vectors

def encode_face_images_arcface(model: InceptionResnetV1, images: Union[np.ndarray, Image.Image], classify = False, device = 'cpu'):
    
    if isinstance(images, Image.Image):
        images = np.array(images)
            
    if len(images.shape) == 3:
        images = images[None, ...]
    
    images = normalize_batch_rgb_images_arcface(images)
    
    model.eval()
    model.classify = classify
        
    images = images.to(device)
    embedding_vectors = model(images)
    
    return embedding_vectors



def encode_face_mtcnn(model, faces: torch.Tensor, classify = False, device = 'cpu', m_type = 'pt') -> torch.Tensor:
                
    if len(faces.shape) == 3:
        faces = faces[None, ...]
        
    if m_type == 'onnx':
        input_name = model.get_inputs()[0].name
        faces = faces.detach().cpu().numpy()
        embedding_vectors = model.run(None, {input_name: faces})
        return torch.from_numpy(embedding_vectors[0])
    
    model.eval()
    model.classify = classify
        
    faces = faces.to(device)
    embedding_vectors = model(faces)
    
    return embedding_vectors

def encode_user_face(model: InceptionResnetV1, user_face_folder_path: str, classify = False):    
    batch_face_images = []
    batch_eye_images = []
    
    for filename in os.listdir(user_face_folder_path):
        filepath = os.path.join(user_face_folder_path, filename)
        img = Image.open(filepath).resize((160, 160))
        img = np.array(img)
        batch_images.append(img)
    
    batch_images = np.array(batch_images)
    
    embedding_vectors = encode_face_images_facenet(model, batch_images, classify)
    
    return embedding_vectors
    
def main(opt, mtcnn, model):
    if os.path.exists(opt.database_path) is False:
        os.mkdir(opt.database_path)
        
    for user_id in os.listdir(opt.img_dir_list):
        
        face_list = []
        eye_list = []
        
        test_face_list = []
        test_eye_list = []
        
        face_embedding_vector_list = []
        eye_embedding_vector_list = []
        
        count = 0
        
        """--------------------------Training------------------------------------""" 
        
        for filename in os.listdir(os.path.join(opt.img_dir_list, user_id, "train")):
            filepath = os.path.join(opt.img_dir_list, user_id, "train", filename)
            
            img = Image.open(filepath)
            
            rgb_frame = np.array(img)
            
            face, box, landmark = detect_user_face(mtcnn, rgb_frame, return_landmarks=True)

            if face is not None:
                if opt.network == 'arcface':
                    face = convert_mtcnn2arcface_norm(face)
                
                eye, rotate_landmark = detect_eye_user(rgb_frame, box, landmark, padding_factor = opt.padding_factor, return_landmark=True, model_name = opt.network)
                
                count += 1
                
                if count <= opt.n_faces:
                    face_list.append(face)
                    eye_list.append(eye)
                    count_batch += 1
                    if count_batch == opt.batch_size:
                        count_batch = 0
                        
                        batch_face = torch.stack(face_list, dim = 0)
                        face_embedding_vectors = encode_face_mtcnn(model, batch_face, device = opt.device)
                        face_list = []
                        face_embedding_vector_list.append(face_embedding_vectors)
                        
                        batch_eye = torch.stack(eye_list, dim = 0)
                        eye_embedding_vectors = encode_face_mtcnn(model, batch_eye, device = opt.device)
                        eye_list = []
                        eye_embedding_vector_list.append(eye_embedding_vectors)
                        
                if count == opt.n_faces:
                    if count_batch != 0:
                        count_batch = 0

                        batch_face = torch.stack(face_list, dim = 0)
                        face_embedding_vectors = encode_face_mtcnn(model, batch_face, device = opt.device)
                        face_list = []
                        face_embedding_vector_list.append(face_embedding_vectors)
                        
                        batch_eye = torch.stack(eye_list, dim = 0)
                        eye_embedding_vectors = encode_face_mtcnn(model, batch_eye, device = opt.device)
                        eye_list = []
                        eye_embedding_vector_list.append(eye_embedding_vectors)
                        
                    face_embedding_vector_list = torch.cat(face_embedding_vector_list, dim = 0)
                    centroid_face_embedding_vectors = clustering_embedding_vectors(face_embedding_vector_list)
                    
                    eye_embedding_vector_list = torch.cat(eye_embedding_vector_list, dim =0)
                    centroid_eye_embedding_vectors = clustering_embedding_vectors(eye_embedding_vector_list)
                    
                    create_new_user(opt.database_path, user_id, user_id, centroid_face_embedding_vectors, mask = opt.mask, eye_embedding_vector = centroid_eye_embedding_vectors)
                    
                    print(face_embedding_vector_list.shape, eye_embedding_vector_list.shape)
                    print("------------Thu thập dữ liệu thành công------------")
                    
                    break

                if opt.n_faces < count <= opt.n_faces + opt.save_n_sample_test:
                    filename = f"{user_id}_{count - opt.n_faces}.jpg"
                    
                    test_face_list.append(face)
                    test_eye_list.append(eye)
                    save_test_image(frame, opt.database_path, user_id, filename)
        
        """--------------------------Testing------------------------------------""" 
                    
        for filename in os.listdir(os.path.join(opt.img_dir_list, user_id, "val")):
            filepath = os.path.join(opt.img_dir_list, user_id, "val", filename)
            
            frame = cv.imread(filepath)
            
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            
            face, box, landmark = detect_user_face(mtcnn, rgb_frame, return_landmarks=True)

            if face is not None:
                if opt.network == 'arcface':
                    face = convert_mtcnn2arcface_norm(face)
                
                eye, rotate_landmark = detect_eye_user(rgb_frame, box, landmark, padding_factor = opt.padding_factor, return_landmark=True, model_name = opt.network)
                
                count += 1
            
                if opt.n_faces < count <= opt.n_faces + opt.save_n_sample_test:
                    filename = f"{user_id}_{count - opt.n_faces}.jpg"
                    
                    test_face_list.append(face)
                    test_eye_list.append(eye)
                    save_test_image(frame, opt.database_path, user_id, filename)
            
                if count == opt.n_faces + opt.save_n_sample_test:
                    test_batch_face = torch.stack(test_face_list, dim=0)
                    face_embedding_vectors = encode_face_mtcnn(model, test_batch_face, device = opt.device)
                    
                    test_batch_eye = torch.stack(test_eye_list, dim = 0)
                    eye_embedding_vectors = encode_face_mtcnn(model, test_batch_eye, device = opt.device)
                    
                    save_test_embedding_vector(face_embedding_vectors, opt.database_path, user_id)
                    save_test_embedding_vector(eye_embedding_vectors, opt.database_path, user_id, mask = True)
                    
                    print("------------Lưu dữ liệu kiểm thử thành công------------")
    

if __name__ == '__main__':
    from create_user import save_embedding_vector, save_test_embedding_vector, save_test_image, create_new_user
    from detect import *
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, default = "facenet", help = "[facenet, arcface]")
    parser.add_argument("--q",  action = 'store_true', help = "quantized model")
    parser.add_argument("--img_dir_list", type=str, default = "database", help = "Database path")
    parser.add_argument("--database_path", type=str, default = "database", help = "Database path")
    parser.add_argument("--n_faces", type=int, default = 90, help = "number of faces is collected")
    parser.add_argument("--batch_size", type=int, default = 10, help = "batch size for encode")
    parser.add_argument("--padding_factor", type=float, default = 0.09375, help = "padding for eye box")
    parser.add_argument("--mask", action = 'store_true', help = "Process mask for face")
    parser.add_argument("--save_n_sample_test", type = int, default = 10, help = "save n images for testing")
    parser.add_argument("--device", type = int, default = -1, help = "-1 for cpu, 0, 1, ... for gpu")
    
    opt = parser.parse_args()
        
    if not torch.cuda.is_available() or opt.device == -1:
        opt.device = 'cpu'

    elif opt.device >= 0:
        opt.device = 'cuda'     
    
    if opt.network == 'facenet':
        mtcnn = MTCNN(image_size = 160, device = opt.device) 
        
        if opt.q:
            opt.device = 'cpu'
            model = Quantized_IResnetv1(pretrained = "vggface2")

            model.eval()
            
            model.fuse_model()
            
            model.qconfig = torch.ao.quantization.get_default_qconfig('x86')

            torch.ao.quantization.prepare(model, inplace=True)

            torch.ao.quantization.convert(model, inplace=True)

            weights = 'weights/quant_vggface2.pt'
        
            model = load_weights(model, weights, device = opt.device).cpu()
        
        else: 
            weights = 'weights/vggface2.pt'
            model = InceptionResnetV1(pretrained = 'vggface2', device = opt.device)
            model = load_weights(model, weights = weights, device = opt.device, eval = False)
        
    else:
        mtcnn = MTCNN(image_size = 112, device = opt.device) 
        weights = 'weights/arcface_weights.pth'
        model = get_model('r50', fp16=False)
        model = load_weights(model, weights = weights, device = opt.device, eval = True)   
        
    main(opt, mtcnn, model)
        