import gradio as gr
from vedo import Volume
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ToTensord
)
from monai.data import Dataset, DataLoader
from monai import first

from radsunet3d import r_a_unet3d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       
model = r_a_unet3d(in_channels=4, out_channels=3, features=[32,64,128,256]).to(device)
checkpoint = torch.load('./model/model.pth', map_location=device, weights_only=False)
pesos = checkpoint['model_state_dict']
model.load_state_dict(pesos)
model.eval()

def proccess(files):
    files = sorted(files)
    print(files)
    transforms = Compose(
            [
                LoadImaged(keys='image'),
                EnsureChannelFirstd(keys='image'),
                NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
                ToTensord(keys='image')
            ]
        )
    tensor = Dataset(data=files, transform=transforms)
    tensor_loader = DataLoader(tensor, batch_size=1)
    return first(tensor_loader)

def predict(tensor):
    with torch.no_grad():
        predict = model(tensor) > 0.5
        predict = torch.sigmoid(predict).float()
    return predict.detach().cpu().numpy()

def view(predict, slide1, slide2, slide3):
    sagital = predict[slide1,0,0].astype(np.float32)
    coronal = predict[0,slide2,0].astype(np.float32)
    axial = predict[0,0,slide3].astype(np.float32)
    return sagital,coronal,axial

def view3d(file,predict, checkb1, checkb2, checkb3):
    # Que el cerebro se vea en 3d y se pueda quitar o poner diferentes secciones del cerebro
    if file is None:
        return None
    
    volumen = Volume(file.name)
    malla = vol.isosurface()
    ruta_temp = "modelo_vedo.obj"
    malla.write(ruta_temp)
    return ruta_temp

def data():
    # Columna para el volumen de cada uno de las secciones del tumor y el volumen del cerebro
    return


def main(files, slide1, slide2, slide3, checkb1, checkb2, checkb3):

    tensor = proccess(files)
    predict = predict(tensor)
    sagital, coronal, axial = view(predict, slide1, slide2, slide3)
    ruta_3d = view3d = view3d(files, predict, checkb1, checkb2, checkb3)
    
    return sagital, coronal, axial, view3d, data

app = gr.Interface(
    fn=main,
    
    inputs=([
        gr.File(label="Upload your files",file_count="multiple"),

        gr.Slider(label="Sagital", minimum=0, maximum=240),
        gr.Slider(label="Coronal", minimum=0, maximum=240),
        gr.Slider(label="Axial", minimum=0, maximum=155),

        gr.Checkbox(value=False, label="Edeme"),
        gr.Checkbox(value=False, label="Nucleo"),
        gr.Checkbox(value=False, label="Realzado"),
    ]),

    outputs=([
        gr.Image(label="Segmentacion Sagital"),
        gr.Image(label="Segmentacion Coronal"),
        gr.Image(label="Segmentacion Axial"),

        gr.Model3D(label="Visualizacion interactiva", clear_color=[1,1,1,1]),

        gr.Dataframe(),
    ]),

    description="Software BraTS2018"
)

app.launch() 
