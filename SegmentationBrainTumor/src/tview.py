import torch
from RADSUNet3D import RadsUNet3D
from torchview import draw_graph

model = RadsUNet3D(in_channels=4, out_channels=3, features=[32,64,128,256])
model.eval()

draw_graph(model, input_size=(1, 4, 128, 128, 128), depth=1,
           device='cpu', save_graph=True, filename='arquitectura_auto')
# genera 'arquitectura_auto.png'