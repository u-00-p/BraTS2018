import gradio as gr
import vedo
import torch

from radsunet3d import r_a_unet3d

"""DEFINICION DEL MODELO"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = r_a_unet3d(in_channels=4, out_channels=3, features=[32,64,128,256]).to(device)
checkpoint = torch.load('./model/model.pth', map_location=device, weights_only=False)
pesos = checkpoint['model_state_dict']
model.load_state_dict(pesos)
model.eval()

"""FUNCIONES"""
def proccess(files):
    files = sorted(files)
    transforms = Compose(
            [
                LoadImaged(keys=['image', 'label']),
                EnsureChannelFirstd(keys='image'),
                ConvertToMultiChannelBasedOnBratsClassesd(keys='label'),
                NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
                CropForegroundd(keys=['image', 'label'], source_key='image'),
                ToTensord(keys=['image', 'label'])
            ]
        )
    tensor = Dataset(data=files, transform=transforms)
    return tensor.unsqueeze(0)

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

def view3d(imagen,predict):
    return

def data():
    return


def main(files, slide1, slide2, slide3, checkb1, checkb2, checkb3):

    tensor = proccess(files)
    predict = predict(tensor)
    sagital, coronal, axial = view(predict, slide1, slide2, slide3)
    view3d = view3d(files, seg, checkb1, checkb2, checkb3)
    
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

        gr.Model3D(),

        gr.Dataframe(),
    ]),

    description="Software BraTS2018"
)

app.launch() 
