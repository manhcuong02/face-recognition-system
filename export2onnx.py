import torch
from facenet import *
from utils import load_weights
import onnx
import numpy as np
import onnxruntime

import time

def export_to_ONNX(model, save_path, weights = None, input_shape = (1, 3, 160, 160)):
    
    for name, param in model.named_parameters():
        param.requires_grad = True
    
    if weights:
        model.load_state_dict(torch.load(weights, map_location='cpu'))
    
    model.eval()
    
    dummy_input = torch.rand(input_shape, requires_grad = True)
    
    torch.onnx.export(
        model = model, 
        args = dummy_input,
        f = save_path,
        export_params = True, # có lưu weights của model vào onnx hay không
        verbose = True,
        do_constant_folding = True, # các weights trong model được tính toán sẵn trong onnx thay vì tính toán lại chúng trong quá trình chạy 
        input_names = ['input'],
        output_names = ['output'],
        dynamic_axes = { # biến thay đổi 
            'input' : {0 : 'batch_size'},    # variable length axes
            'output' : {0 : 'batch_size'}
        }
    )



if __name__ == '__main__':
    # weights = 'weights/vggface2.pt'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = InceptionResnetV1(pretrained= "vggface2", device = device)    
    
    # export_to_ONNX(model, "weights/iresnetv1.onnx", weights, (1, 3, 160, 160))
    
    MTCNN
    # # kiểm tra cấu trúc và hợp lệ của mô hình ONNX đã được tạo ra bằng cách xuất từ PyTorch
    onnx_path = "weights/iresnetv1.onnx"
    # onnx_model = onnx.load(onnx_path)
    # onnx.checker.check_model(onnx_model)


    onnxruntime_model = onnxruntime.InferenceSession(onnx_path)
    input_name = onnxruntime_model.get_inputs()[0].name

    start_time = time.time()
        
    x = np.random.rand(1, 3, 160, 160).astype(np.float32)
    out = onnxruntime_model.run(None, {input_name: x})
    
    print(out[0].shape)
    
    