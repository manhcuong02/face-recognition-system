from .backbones import get_model

name = "r50"
fp16 = False

arcface_model = get_model(name, fp16=False)

__all__ = ['arcface_model', 'get_model']
