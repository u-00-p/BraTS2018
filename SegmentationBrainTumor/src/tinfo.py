import torch
from RADSUNet3D import RadsUNet3D
from torchinfo import summary

model = RadsUNet3D(in_channels=4, out_channels=3, features=[32,64,128,256])
model.eval()

summary(model, input_size=(1, 4, 128, 128, 128), depth=2,
        col_names=["output_size", "num_params"])