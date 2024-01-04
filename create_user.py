import os
import torch
import cv2 as cv
import numpy as np
from encoder import *
from utils.clustering import *
from utils.plot import *
import argparse
from facenet import MTCNN, InceptionResnetV1
from recognition import *
from arcface import get_model

def create_new_user_id(database_path, user_id):
    user_id_path = os.path.join(database_path, user_id)
    
    if not os.path.exists(user_id_path):
        os.mkdir(user_id_path)

def save_user_name(user_id_path, user_name):
    user_name_path = os.path.join(user_id_path, "name.txt")
    
    file = open(user_name_path, "w", encoding= 'utf-8')
    file.write(user_name)
     
def save_embedding_vector(embedding_vector: Tuple[np.ndarray, torch.Tensor], database_path, user_id: str, mask = False)->np.ndarray:
    if isinstance(embedding_vector, torch.Tensor):
        embedding_vector = embedding_vector.detach().cpu().numpy()
        
    user_id_path = os.path.join(database_path, user_id)
    
    if not os.path.exists(user_id_path):
        os.mkdir(user_id_path)

    if mask is True:
        embedding_vector_path = os.path.join(user_id_path, "eye_vector.txt")
    else: 
        embedding_vector_path = os.path.join(user_id_path, "face_vector.txt")

    np.savetxt(embedding_vector_path, embedding_vector, fmt = "%.5f", delimiter = ' ')
     
def create_new_user(database_path, user_id, user_name, face_embedding_vectors, mask = False, eye_embedding_vector = None):
    create_new_user_id(database_path, user_id)
    user_id_path = os.path.join(database_path, user_id)
    save_user_name(user_id_path, user_name)
    save_embedding_vector(face_embedding_vectors, database_path, user_id)
    if mask and eye_embedding_vector is not None: 
        save_embedding_vector(eye_embedding_vector, database_path, user_id, mask = True)
                
def save_embedding_user(model, database_path,  user_id, mask = False):
    
    user_face_folder_path = os.path.join(database_path, user_id)
    
    embedding_vectors = encode_user_face(model, user_face_folder_path)
    
    centroid_embedding_vectors = clustering_embedding_vectors(embedding_vectors)
        
    save_embedding_vector(centroid_embedding_vectors, database_path, user_id, mask = mask)

def save_test_image(image, database_path, user_id, filename):
    test_image_dir = os.path.join(database_path, user_id, "test_images")

    if not os.path.exists(test_image_dir):
        os.mkdir(test_image_dir)
    
    file_path = os.path.join(test_image_dir, filename)    
    
    cv.imwrite(file_path, image)

def save_test_embedding_vector(embedding_vector: Tuple[np.ndarray, torch.Tensor], database_path, user_id: str, mask = False)->np.ndarray:
    if isinstance(embedding_vector, torch.Tensor):
        embedding_vector = embedding_vector.detach().cpu().numpy()
        
    user_id_path = os.path.join(database_path, user_id)
    
    if not os.path.exists(user_id_path):
        os.mkdir(user_id_path)

    if mask is True:
        embedding_vector_path = os.path.join(user_id_path, "test_eye_vector.txt")
    else: 
        embedding_vector_path = os.path.join(user_id_path, "test_face_vector.txt")

    np.savetxt(embedding_vector_path, embedding_vector, fmt = "%.5f", delimiter = ' ')

def main(opt ,mtcnn: MTCNN, resnet: InceptionResnetV1):
    user_name = opt.name[0]
    user_id = opt.user_id[0]
    
    if opt.batch_size > opt.n_faces:
        opt.batch_size = opt.n_faces
    
    if opt.source == "0":
        video = cv.VideoCapture(0)
    
    else:
        video = cv.VideoCapture(opt.source)
    count = 0
    
    count_batch = 0
    
    face_list = []
    eye_list = []
    
    face_embedding_vector_list = []
    eye_embedding_vector_list = []
    
    test_face_list = []
    test_eye_list = []
    
    print("------------Đang thu thập dữ liệu------------")
    while True:
        ret, frame = video.read() # read
        
        if ret:
            if opt.source == "0":
                frame = cv.flip(frame, 1)
            
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            
            face, box, landmark = detect_user_face(mtcnn, rgb_frame, return_landmarks=True)

            if face is not None:
                if opt.network == 'arcface':
                    face = convert_mtcnn2arcface_norm(face)
                
                eye, rotate_landmark = detect_eye_user(rgb_frame, box, landmark, padding_factor = opt.padding_factor, return_landmark=True, model_name = opt.network)
                
                front = detect_orientation(rotate_landmark)
                
                if front is False:
                    continue
                
                count += 1
                
                if count <= opt.n_faces:
                    face_list.append(face)
                    eye_list.append(eye)
                    count_batch += 1
                    if count_batch == opt.batch_size:
                        count_batch = 0
                        
                        batch_face = torch.stack(face_list, dim = 0)
                        face_embedding_vectors = encode_face_mtcnn(resnet, batch_face, device = opt.device, m_type = opt.m_type)
                        face_list = []
                        face_embedding_vector_list.append(face_embedding_vectors)
                        
                        batch_eye = torch.stack(eye_list, dim = 0)
                        eye_embedding_vectors = encode_face_mtcnn(resnet, batch_eye, device = opt.device, m_type = opt.m_type)
                        eye_list = []
                        eye_embedding_vector_list.append(eye_embedding_vectors)
                        
                if count == opt.n_faces:
                    if count_batch != 0:
                        count_batch = 0

                        batch_face = torch.stack(face_list, dim = 0)
                        face_embedding_vectors = encode_face_mtcnn(resnet, batch_face, device = opt.device, m_type = opt.m_type)
                        face_list = []
                        face_embedding_vector_list.append(face_embedding_vectors)
                        
                        batch_eye = torch.stack(eye_list, dim = 0)
                        eye_embedding_vectors = encode_face_mtcnn(resnet, batch_eye, device = opt.device, m_type = opt.m_type)
                        eye_list = []
                        eye_embedding_vector_list.append(eye_embedding_vectors)
                        
                    face_embedding_vector_list = torch.cat(face_embedding_vector_list, dim = 0)
                    centroid_face_embedding_vectors = clustering_embedding_vectors(face_embedding_vector_list)
                    
                    eye_embedding_vector_list = torch.cat(eye_embedding_vector_list, dim =0)
                    centroid_eye_embedding_vectors = clustering_embedding_vectors(eye_embedding_vector_list)
                    
                    create_new_user(opt.database_path, user_id, user_name, centroid_face_embedding_vectors, mask = opt.mask, eye_embedding_vector = centroid_eye_embedding_vectors)
                    
                    print(face_embedding_vector_list.shape, eye_embedding_vector_list.shape)
                    print("------------Thu thập dữ liệu thành công------------")

                if opt.n_faces < count <= opt.n_faces + opt.save_n_sample_test:
                    filename = f"{user_id}_{count - opt.n_faces}.jpg"
                    
                    test_face_list.append(face)
                    test_eye_list.append(eye)
                    save_test_image(frame, opt.database_path, user_id, filename)
                    
                if count == opt.n_faces + opt.save_n_sample_test:
                    test_batch_face = torch.stack(test_face_list, dim=0)
                    face_embedding_vectors = encode_face_mtcnn(resnet, test_batch_face, device = opt.device, m_type = opt.m_type)
                    
                    test_batch_eye = torch.stack(test_eye_list, dim = 0)
                    eye_embedding_vectors = encode_face_mtcnn(resnet, test_batch_eye, device = opt.device, m_type = opt.m_type)
                    
                    save_test_embedding_vector(face_embedding_vectors, opt.database_path, user_id)
                    save_test_embedding_vector(eye_embedding_vectors, opt.database_path, user_id, mask = True)
                    
                    print("------------Lưu dữ liệu kiểm thử thành công------------")

                    if opt.source != "0":
                        break

            if opt.source == "0":
                frame = plot_box_and_label(frame, lw = 1, box = box, color = (0,0,255), landmark=landmark)
                cv.imshow("Collect data", frame)
                if cv.waitKey(10) & 0xFF == ord("s"):
                    break
    
    video.release()
    cv.destroyAllWindows()
    

if __name__ == '__main__':    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, default = "facenet", help = "[facenet, arcface]")
    parser.add_argument("--m_type",  type = str,default = "onnx",  help = "model type in [pt, quant, onnx] of facenet")
    parser.add_argument("--source", type=str, default = "0", help = "Video or webcam")
    parser.add_argument("--database_path", type=str, default = "onnx_database", help = "Database path")
    parser.add_argument("--user_id", type=str, nargs="+", help="id of the new user")
    parser.add_argument("--name", type=str, nargs="+", default="", help="name of the new user")
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
        assert opt.m_type in ['pt', 'quant', 'onnx']
        
        mtcnn = MTCNN(image_size = 160, device = opt.device) 
        
        if opt.m_type == "quant":
            from facenet import Quantized_IResnetv1
            
            opt.device = 'cpu'
            model = Quantized_IResnetv1(pretrained = "vggface2")

            model.eval()
            
            model.fuse_model()
            
            model.qconfig = torch.ao.quantization.get_default_qconfig('x86')

            torch.ao.quantization.prepare(model, inplace=True)

            torch.ao.quantization.convert(model, inplace=True)

            weights = 'weights/quant_vggface2.pt'
        
            model = load_weights(model, weights, device = opt.device).cpu()
        
        elif opt.m_type == 'onnx':
            import onnxruntime
            
            opt.device = 'cpu'
            onnx_path = "weights/iresnetv1.onnx"
            model = onnxruntime.InferenceSession(onnx_path)
        
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

