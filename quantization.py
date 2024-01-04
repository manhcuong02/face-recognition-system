import cv2 as cv 
from utils import load_weights
import torch
from detect import *
import numpy as np
from PIL import Image

from facenet import Quantized_IResnetv1, InceptionResnetV1, MTCNN
from encoder import *

# Set up warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)

# Set up warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)

# Specify random seed for repeatable results
torch.manual_seed(191009)
import time

if __name__ == "__main__":

    mtcnn_model = MTCNN(device = 'cpu', margin = 60)

    img = Image.open("images/trang2.jpg").convert("RGB")

    img = np.array(img)
    
    face, box, landmark = detect_user_face(mtcnn_model, img, return_landmarks=True)

    model = Quantized_IResnetv1(pretrained = "vggface2")
    
    model = load_weights(model, "weights/vggface2.pt").cpu()
    model.eval()
    
    model.fuse_model()
    
    model.qconfig = torch.ao.quantization.get_default_qconfig('x86')

    print(model.qconfig)
    torch.quantization.prepare(model, inplace=True)

    print('Post Training Quantization Prepare: Inserting Observers')
    print('\n After observer insertion \n\n', model.conv2d_1a.conv)

    
    encode_face_mtcnn(model, face)

    torch.ao.quantization.convert(model, inplace=True)
    
    print('Post Training Quantization: Convert done')
    print('\n Block: After fusion and quantization, note fused modules: \n\n',model.conv2d_1a.conv)

    torch.save(model.state_dict(), "weights/quant_vggface2.pt")    
    
