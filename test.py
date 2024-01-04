import cv2 as cv
# from create_user import *
from facenet import MTCNN, InceptionResnetV1, Quantized_IResnetv1
import torch
from utils import *
import math
import numpy as np
import time
# model = InceptionResnetV1(pretrained= 'vggface2', device = 'cpu')

# model = load_weights(model, weights = 'weights/vggface2.pt')
# model.eval()
# x = torch.rand(3, 3, 160, 160)

# print(model(x))

model = Quantized_IResnetv1(pretrained = "vggface2")
    
model.eval()

model.fuse_model()

model.qconfig = torch.ao.quantization.get_default_qconfig('x86')

torch.ao.quantization.prepare(model, inplace=True)

torch.ao.quantization.convert(model, inplace=True)

model = load_weights(model, weights = 'weights/quant_vggface2.pt', eval = False).cpu()

start_time = time.time()

for i in range(100):
    x = torch.rand(3, 3, 160, 160)
    y = model(x)

end_time = time.time()

print("Time:", (end_time - start_time)*1000)