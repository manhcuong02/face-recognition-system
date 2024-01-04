import torch 
from facenet import MTCNN, InceptionResnetV1, Quantized_IResnetv1
import cv2 as cv
import os
import numpy as np
from utils.plot import *
from utils.load_user_infomation import load_database
from utils.distance import *


from encoder import *
import time
from detect import *
import argparse
from queue import Queue
import threading
from arcface import get_model

def recognize_user_face(face: torch.Tensor, eye: torch.Tensor, model: InceptionResnetV1,
                        user_id_list: Union[np.ndarray, List, torch.Tensor],
                        name_list: Union[np.ndarray, List, torch.Tensor],
                        face_embedding_list: Union[np.ndarray, List, torch.Tensor],
                        eye_embedding_list: Union[np.ndarray, List, torch.Tensor] = None, device = 'cpu', m_type = 'pt'):
    
    input_stack = torch.stack([face, eye], dim = 0)

    embedding_vector = encode_face_mtcnn(model, input_stack, device = device, m_type = m_type)
    
    face_embedding_vector = embedding_vector[0].reshape(1, -1)
    eye_embedding_vector = embedding_vector[1].reshape(1, -1)
    
    face_dist, face_idx= cosine_similarity(face_embedding_list, face_embedding_vector)

    print("--------------------------------")
    print(f"Name: {name_list[face_idx]}, Dist: {face_dist: .5f}")
    
    if eye_embedding_list is not None:
        eye_dist, eye_idx = cosine_similarity(eye_embedding_list, eye_embedding_vector)
        
        print(f"Name: {name_list[eye_idx]}, Dist: {eye_dist: .5f}")
        
        if eye_dist < face_dist:
            face_idx = eye_idx
    
    return user_id_list[face_idx], name_list[face_idx]
    
def display_image(img_queue: Queue, result_queue: Queue):
    
    global flag
    
    video = cv.VideoCapture(0)
    
    while True:
        ret, frame = video.read() # read
        if ret:
            frame = cv.flip(frame, 1)
            
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            
            if img_queue.qsize() == 1:
                _ = img_queue.get()
            img_queue.put(rgb_frame)
                
            if not result_queue.empty():
                result = result_queue.get()
                result_queue.put(result)
            else:
                result = [None, None]
                
            frame = plot_box_and_label(frame, box = result[0], color = (0, 0, 255), label = result[1])    
            
            cv.imshow("Recognize", frame)            
            
            if cv.waitKey(10) & 0xFF == ord("s"):
                flag = True
                break
    
    video.release()
    cv.destroyAllWindows()  
    
def main(opt, mtcnn: MTCNN, resnet: InceptionResnetV1, img_queue: Queue, result_queue: Queue):
    
    user_id_list, name_list, face_embedding_list, eye_embedding_list = load_database(opt.database_path, opt.mask)
    
    global flag
    
    while True:
        if not img_queue.empty():
            rgb_frame = img_queue.get()
                        
            face, box, landmark = detect_user_face(mtcnn, rgb_frame, return_landmarks = True)
            if face is not None:
                if opt.network == 'arcface':
                    face = convert_mtcnn2arcface_norm(face)
                eye = detect_eye_user(rgb_frame, box, landmark, opt.padding_factor, model_name = opt.network)
                user_id, user_name = recognize_user_face(face, eye, resnet, user_id_list, name_list, face_embedding_list, eye_embedding_list, device = opt.device, m_type = opt.m_type)
                
            if result_queue.qsize() == 1:
                _ = result_queue.get()
            
            result_queue.put([box, user_name])
            
            if flag == True:
                break
 

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", type=str, default = "onnx_database", help = "Database path")
    parser.add_argument("--padding_factor", type=float, default = 0.09375, help = "padding for eye box")
    parser.add_argument("--mask", action = 'store_true', help = "Recognition with mask")
    parser.add_argument("--network", type=str, default = "facenet", help = "[facenet, arcface]")
    parser.add_argument("--m_type",  type = str,default = "onnx",  help = "model type in [pt, quant, onnx] of facenet")
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
        
    
    global flag
    flag = False
    
    image_queue = Queue(maxsize=1)  # Kích thước hàng đợi là 1
    results = Queue(maxsize=1)  # Kích thước hàng đợi là 1

    # Tạo và khởi động luồng hiển thị ảnh
    display_thread = threading.Thread(target=display_image, args=(image_queue,results))
    display_thread.start()

    # Tạo và khởi động luồng xử lý ảnh
    process_thread = threading.Thread(target=main, args=(opt, mtcnn, model, image_queue, results))
    process_thread.start()

    # Chờ cả hai luồng hoàn thành
    display_thread.join()
    process_thread.join()
    