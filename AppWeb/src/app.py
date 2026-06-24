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

    # Canal 1 del tensor = T1ce ya normalizado (rango aprox -1 a 3)
    # Lo guardamos como fondo gris para los cortes 2D.
    # Por qué usar el tensor normalizado y no el NIfTI crudo:
    #   el tensor ya pasó por NormalizeIntensityd, así que las
    #   intensidades son comparables entre pacientes. El NIfTI
    #   crudo varía mucho de paciente a paciente.
    fondo_2d = tensor[0, 1].cpu().numpy()  # shape [X, Y, Z]

    # Para el 3D necesitamos el NIfTI crudo (intensidades reales)
    # porque vedo.isosurface necesita un umbral de intensidad.
    vol_cerebro = nib.load(ruta_t1ce).get_fdata().astype(np.float32)

    return resultado_np, fondo_2d, vol_cerebro

def actualizar_2d(resultado_np, fondo_2d, s1, s2, s3):
    if resultado_np is None:
        return None, None, None

    vol = resultado_np[0]   # [3, X, Y, Z]  canales: TC, WT, ET
    s1, s2, s3 = int(s1), int(s2), int(s3)

    # Clampear al rango válido de índices
    s1 = min(s1, vol.shape[1] - 1)
    s2 = min(s2, vol.shape[2] - 1)
    s3 = min(s3, vol.shape[3] - 1)

    def hacer_corte(plano_tumor, corte_cerebro):
        """
        plano_tumor:   [3, H, W]  — máscara binaria de TC, WT, ET
        corte_cerebro: [H, W]     — intensidad T1ce normalizada

        Estrategia de color:
          PASO 1 — fondo gris: el cerebro real, normalizado a [0,1].
                   Así el médico ve la anatomía completa.
          PASO 2 — tumor encima: donde hay tumor, sobreescribimos el
                   gris con el color de la región correspondiente.
                   Usamos colores puros y distintos para que no se
                   confundan aunque las regiones se solapan.

        Colores elegidos:
          Edema (WT - TC) → azul claro   (0.3, 0.6, 1.0)
          TC (núcleo)     → rojo-naranja (1.0, 0.3, 0.1)
          ET (realce)     → amarillo     (1.0, 1.0, 0.0)
          Pintamos en ese orden: primero edema, luego TC encima,
          luego ET encima del TC — así el más pequeño (ET) siempre
          es visible aunque esté dentro del TC.
        """
        H, W = corte_cerebro.shape

        # PASO 1: fondo gris normalizado al percentil 1-99
        # (evita que un vóxel muy brillante aplaste todo lo demás)
        vmin = np.percentile(corte_cerebro, 1)
        vmax = np.percentile(corte_cerebro, 99)
        gris = np.clip((corte_cerebro - vmin) / (vmax - vmin + 1e-8), 0, 1)

        rgb = np.stack([gris, gris, gris], axis=-1)  # [H, W, 3] gris

        # PASO 2: pintar tumor encima del gris
        edema = (plano_tumor[1] - plano_tumor[0]) > 0.5  # WT - TC = edema
        tc    = plano_tumor[0] > 0.5
        et    = plano_tumor[2] > 0.5

        rgb[edema] = [0.2, 0.5, 1.0]   # azul claro  → edema
        rgb[tc]    = [1.0, 0.3, 0.1]   # rojo-naranja → TC
        rgb[et]    = [1.0, 1.0, 0.0]   # amarillo     → ET

        return rgb.astype(np.float32)

    return (
        hacer_corte(vol[:, s1, :, :], fondo_2d[s1, :, :]),
        hacer_corte(vol[:, :, s2, :], fondo_2d[:, s2, :]),
        hacer_corte(vol[:, :, :, s3], fondo_2d[:, :, s3]),
    )

def actualizar_3d(resultado_np, vol_cerebro, show_tc, show_wt, show_et, show_brain):
    if resultado_np is None:
        return None

    capas = [
        # Umbral relativo: percentil 30 de los vóxeles no-cero.
        # Por qué no un número fijo como 300:
        #   las intensidades BraTS varían por paciente y por escáner.
        #   Un umbral fijo que funciona para un paciente falla en otro.
        #   El percentil 30 siempre captura la misma "fracción" del
        #   tejido, independientemente de la escala de intensidades.
        ('cerebro', vol_cerebro,
         [200, 200, 200, 25], show_brain,
         float(np.percentile(vol_cerebro[vol_cerebro > 0], 30))),
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
    state_resultado = gr.State(value=None)   # numpy [1,3,X,Y,Z]  — máscara
    state_fondo     = gr.State(value=None)   # numpy [X,Y,Z]      — T1ce normalizado
    state_cerebro   = gr.State(value=None)   # numpy [X,Y,Z]      — T1ce crudo para 3D

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
        resultado_np, fondo_2d, vol_cerebro = predecir(files)
        sag, cor, axi = actualizar_2d(resultado_np, fondo_2d, s1, s2, s3)
        glb           = actualizar_3d(resultado_np, vol_cerebro, tc, wt, et, brain)
        tabla_df      = hacer_tabla(resultado_np)
        return resultado_np, fondo_2d, vol_cerebro, sag, cor, axi, glb, tabla_df

    btn_predecir.click(
        fn=on_predecir,
        inputs=[archivos, sl_sag, sl_cor, sl_axi, cb_tc, cb_wt, cb_et, cb_brain],
        outputs=[state_resultado, state_fondo, state_cerebro,
                 img_sag, img_cor, img_axi, modelo_3d, tabla],
    )

    def on_slider(resultado_np, fondo_2d, s1, s2, s3):
        return actualizar_2d(resultado_np, fondo_2d, s1, s2, s3)

    for slider in [sl_sag, sl_cor, sl_axi]:
        slider.change(
            fn=on_slider,
            inputs=[state_resultado, state_fondo, sl_sag, sl_cor, sl_axi],
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
