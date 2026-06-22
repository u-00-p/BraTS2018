import gradio as gr
import vedo


def predict():
    return

def view():
    return

def view3d(imagen,mask):
    return

def data():
    return


def main(files, slide1, slide2, slide3, checkb1, checkb2, checkb3):
    seg = predict(files)
    
    sagital, coronal, axial = view(files, seg, slide1, slide2, slide3)
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
