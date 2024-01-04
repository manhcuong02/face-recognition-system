from flask import Flask, render_template, Response, request, jsonify
import cv2
from utils.load_user_infomation import load_database
from facenet import MTCNN, InceptionResnetV1
from utils import load_weights
import torch
from detect import *
from encoder import *
from create_user import create_new_user
from PIL import Image
from recognition import recognize_user_face
from arcface import get_model
import io
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--database_path", type=str, default = "onnx_database", help = "Database path")
parser.add_argument("--padding_factor", type=float, default = 0.09375, help = "padding for eye box")
parser.add_argument("--mask", action = 'store_true', help = "Recognition with mask")
parser.add_argument("--network", type=str, default = "facenet", help = "[facenet, arcface]")
parser.add_argument("--m_type",  type = str,default = "pt",  help = "model type in [pt, quant, onnx] of facenet")
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

user_id_list, name_list, face_embedding_list, eye_embedding_list = load_database(opt.database_path, mask = opt.mask)

app = Flask(__name__)

@app.route("/")
def home():
    return "Trang chủ"

@app.route("/create_user", methods=["POST"])
def create_user_function():
    if request.method == "POST":
        user_name = request.form.get("user_name")
        user_id = request.form.get("user_id")
        
        img_filepath = request.files.get("image")
        
        binary_img = img_filepath.read()
        img = Image.open(io.BytesIO(binary_img))
              
        rgb_image = np.array(img)
        
        face, box, landmark = detect_user_face(mtcnn, rgb_image, return_landmarks=True)
        
        if face is not None:
            if opt.network == "arcface":
                face = convert_mtcnn2arcface_norm(face)
            
            eye = detect_eye_user(rgb_image, box, landmark, model_name = opt.network, padding_factor= opt.padding_factor)
            face_embedding_vectors = encode_face_mtcnn(model, face, device = opt.device, m_type = opt.m_type)
            eye_embedding_vectors = encode_face_mtcnn(model, eye, device = opt.device, m_type = opt.m_type)
            
            create_new_user(opt.database_path, user_id, user_name, face_embedding_vectors.reshape(-1), eye_embedding_vector = eye_embedding_vectors.reshape(-1), mask = opt.mask)
                    
        return f"Lưu dữ liệu người dùng: {user_name} có user_id: {user_id} thành công"
        
    return "Quá trình lưu dữ liệu người dùng bị lỗi"
    
@app.route("/user_recognition", methods=["POST"])
def user_recognition_function(): 
    if request.method == "POST":
        img_filepath = request.files.get("image")
        img = Image.open(img_filepath)
        rgb_image = np.array(img)
        
        face, box, landmark = detect_user_face(mtcnn, rgb_image, return_landmarks=True)
        
        if face is not None:
            if opt.network == "arcface":
                face = convert_mtcnn2arcface_norm(face)
            
            eye = detect_eye_user(rgb_image, box, landmark, model_name=opt.network, padding_factor= opt.padding_factor)
            user_id, user_name = recognize_user_face(face, eye, model, user_id_list, name_list, face_embedding_list, eye_embedding_list, device = opt.device, m_type = opt.m_type)

            return str({
                "bbox": box,
                "user_id": user_id,
                "user_name": user_name
            })
        
        return str({
            "bbox": None,
            "user_id": None,
            "user_name": None
        })
    return 
        
        
if __name__=='__main__':
    app.run(debug=True, host = '0.0.0.0', port = 6868)
    
    