import gradio as gr
from vedo import Volume
import torch
import pandas as pd
import os
import trimesh
import nibabel as nib

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    ToTensord,
    ConcatItemsd
)
from monai.data import Dataset, DataLoader
from monai.utils import first
from monai.inferers import SlidingWindowInferer

import numpy as np

from radsunet3d import r_a_unet3d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       
model = r_a_unet3d(in_channels=4, out_channels=3, features=[32,64,128,256]).to(device)
checkpoint = torch.load('./model/model.pth', map_location=device, weights_only=False)
pesos = checkpoint['model_state_dict']
model.load_state_dict(pesos)
model.eval()

inferidor = SlidingWindowInferer(roi_size=(128,128,128), sw_batch_size=1, overlap=0.5)

def predecir(files):
    if not files:
        raise gr.Error("Sube los 4 archivos .nii.gz")

    data_dict = {}
    ruta_t1ce = None
    
    for f in files:
        nm = f.name
        if '_t1ce' in nm:
            data_dict['t1ce'] = nm
            ruta_t1ce = nm
        elif '_t1' in nm:
            data_dict['t1'] = nm
        elif '_t2' in nm:
            data_dict['t2'] = nm
        elif '_flair' in nm:
            data_dict['flair'] = nm
    
    faltantes = [m for m in ['flair', 't1ce', 't1', 't2'] if m not in data_dict]
    if faltantes:
        raise gr.Error(f'Faltan modalidades: {faltantes}')

    tf = Compose(
        [
            LoadImaged(keys=["flair", "t1ce", "t1", "t2"]),
            EnsureChannelFirstd(keys=["flair", "t1ce", "t1", "t2"]),
            ConcatItemsd(keys=["flair", "t1ce", "t1", "t2"], name='image', dim=0),
            NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        ]
    )

    tensor = tf(data_dict)['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = inferidor(inputs=tensor, network=model)
        resultado = (torch.sigmoid(logits) > 0.5).float()
    resultado_np = resultado.detach().cpu().numpy()
    vol_cerebro = nib.load(ruta_t1ce).get_fdata().astype(np.float32)

    return resultado_np, vol_cerebro

def actualizar_2d(resultado_np, s1,s2,s3):
    if resultado_np is None:
        return None,None,None

    vol = resultado_np[0]
    s1,s2,s3 = int(s1), int(s2), int(s3)

    s1 = min(s1, vol.shape[1]-1)
    s2 = min(s2, vol.shape[2]-1)
    s3 = min(s3, vol.shape[3]-1)



    def rgb(plano):
        H,W = plano.shape[1], plano.shape[2]
        out = np.zeros((H,W,3), dtype=np.float32)
        out[...,0] = plano[0]
        out[...,2] = np.clip(plano[1] - plano[0], 0 , 1)
        out[...,1] = plano[2]
        return out

    return (
        rgb(vol[:,s1,:,:]),
        rgb(vol[:,:,s2,:]),
        rgb(vol[:,:,:,s3])
    )

def actualizar_3d(resultado_np, vol_cerebro, show_tc, show_wt, show_et, show_brain):
    if resultado_np is None:
        return None

    capas = [
        ('cerebro', vol_cerebro,          [200,200,200, 25], show_brain,  300.0),
        ('TC',      resultado_np[0, 0],   [220, 50, 50,180], show_tc,      0.5),
        ('WT',      resultado_np[0, 1],   [ 50,200, 50, 90], show_wt,      0.5),
        ('ET',      resultado_np[0, 2],   [255,230,  0,255], show_et,      0.5),
    ]

    piezas = []

    for nombre, arr, color, activo, umbral in capas:
        if not activo:
            continue

        a = np.ascontiguousarray(arr.astype(np.float32))
        if a.max() < umbral:
            print(f"{nombre} vacia")
            continue

        malla = Volume(a).isosurface(value=umbral)

        if nombre == "cerebro":
            malla = malla.decimate(fraction=0.04)

        if malla.npoints == 0:
            print(f"{nombre} isosurface vacia")
            continue

        verts = np.array(malla.vertices)
        caras = np.array(malla.cells)

        tm = trimesh.Trimesh(vertices=verts, faces=caras, process=False)
        tm.visual.vertex_colors = np.tile(
            np.array(color, dtype=np.uint8), (len(verts), 1)
        )
        piezas.append(tm)
        print(f"  {nombre}: {len(verts)} vértices, {len(caras)} caras")

    if not piezas:
        return None

    escena   = trimesh.Scene(piezas)
    ruta_glb = os.path.abspath("tumor.glb")
    escena.export(ruta_glb)
    print(f"  .glb")
    return ruta_glb
        
def hacer_tabla(resultado_np):
    if resultado_np is None:
        return None

    return pd.DataFrame({
        "Región":   ["TC — Tumor Core", "WT — Whole Tumor", "ET — Enhancing"],
        "Color 3D": ["Rojo",             "Verde",             "Amarillo"],
        "Vóxeles":  [int(resultado_np[0,i].sum()) for i in range(3)],
    })



with gr.Blocks(title="Segmentador BraTS") as app:
    state_resultado = gr.State(value=None)
    state_cerebro = gr.State(value=None)

    gr.Markdown("# Segmentador de tumores cerebrales — BraTS 2018")
    gr.Markdown("**Paso 1:** sube los 4 archivos. **Paso 2:** pulsa *Predecir*. **Paso 3:** ajusta los controles.")

    with gr.Row():
        with gr.Column(scale=1):
            archivos = gr.File(
                label="Sube los 4 .nii.gz (flair, t1, t1ce, t2)",
                file_count="multiple"
            )
            btn_predecir = gr.Button("Predecir", variant="primary")
            tabla = gr.Dataframe(label="Volumen por región (vóxeles)")
 
        with gr.Column(scale=1):
            gr.Markdown("### Cortes 2D")
            sl_sag = gr.Slider(0, 239, value=120, step=1, label="Sagital (X)")
            sl_cor = gr.Slider(0, 239, value=120, step=1, label="Coronal (Y)")
            sl_axi = gr.Slider(0, 154, value=77,  step=1, label="Axial (Z)")
 
            gr.Markdown("### Capas 3D")
            cb_tc    = gr.Checkbox(value=True,  label="TC — Tumor Core (rojo)")
            cb_wt    = gr.Checkbox(value=True,  label="WT — Whole Tumor (verde)")
            cb_et    = gr.Checkbox(value=True,  label="ET — Enhancing (amarillo)")
            cb_brain = gr.Checkbox(value=False, label="Cerebro completo (gris, semitransparente)")

    with gr.Row():
        img_sag = gr.Image(label="Sagital")
        img_cor = gr.Image(label="Coronal")
        img_axi = gr.Image(label="Axial")
 
    modelo_3d = gr.Model3D(
        label="Visualización 3D interactiva",
        clear_color=[0.12, 0.12, 0.12, 1]
    )

    def on_predecir(files, s1, s2, s3, tc, wt, et, brain):
        resultado_np, vol_cerebro = predecir(files)
        sag, cor, axi = actualizar_2d(resultado_np, s1, s2, s3)
        glb           = actualizar_3d(resultado_np, vol_cerebro, tc, wt, et, brain)
        tabla_df      = hacer_tabla(resultado_np)
        return resultado_np, vol_cerebro, sag, cor, axi, glb, tabla_df
 
    btn_predecir.click(
        fn=on_predecir,
        inputs=[archivos, sl_sag, sl_cor, sl_axi, cb_tc, cb_wt, cb_et, cb_brain],
        outputs=[state_resultado, state_cerebro,
                 img_sag, img_cor, img_axi, modelo_3d, tabla],
    )

    def on_slider(resultado_np, s1, s2, s3):
        return actualizar_2d(resultado_np, s1, s2, s3)
 
    for slider in [sl_sag, sl_cor, sl_axi]:
        slider.change(
            fn=on_slider,
            inputs=[state_resultado, sl_sag, sl_cor, sl_axi],
            outputs=[img_sag, img_cor, img_axi],
        )

    def on_checkbox(resultado_np, vol_cerebro, tc, wt, et, brain):
        return actualizar_3d(resultado_np, vol_cerebro, tc, wt, et, brain)
 
    for cb in [cb_tc, cb_wt, cb_et, cb_brain]:
        cb.change(
            fn=on_checkbox,
            inputs=[state_resultado, state_cerebro, cb_tc, cb_wt, cb_et, cb_brain],
            outputs=[modelo_3d],
        )
 
app.launch()

    

