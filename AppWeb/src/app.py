import os
import gradio as gr
import numpy as np
import pandas as pd
import torch
import joblib
import trimesh
import nibabel as nib
from vedo import Volume

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    ConcatItemsd,
)
from monai.inferers import SlidingWindowInferer

from radsunet3d import r_a_unet3d

# -----------------------------------------------------------------------------
# Configuración
# -----------------------------------------------------------------------------
MODELO_SEG = './model/model_brats.pth'
MODELO_SUPERVIVENCIA = './model/survival_model.joblib'
ROI = (128, 128, 128)
MODALIDADES = ['flair', 't1ce', 't1', 't2']

# Orden EXACTO de las 12 variables predictoras con las que se entrenó el .joblib
# (columnas de 'new_features.csv' tras quitar Survival, File_name, Survival_time
#  y ET_quadrant).
FEATURES_MODELO = [
    "Age", "WT_quadrant", "TC_quadrant", "WT_volume", "NCR/NET_volume",
    "TC_volume", "ET_volume", "ED_volume", "Distance_mid_brain-tumor",
    "healthy_br_vol", "tumor_percen", "healthy_br_percen",
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

model = r_a_unet3d(in_channels=4, out_channels=3, features=[32, 64, 128, 256]).to(device)
checkpoint = torch.load(MODELO_SEG, map_location=device, weights_only=False)
pesos = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
model.load_state_dict(pesos)
model.eval()

inferidor = SlidingWindowInferer(roi_size=ROI, sw_batch_size=2, overlap=0.5)

tf = Compose([
    LoadImaged(keys=MODALIDADES),
    EnsureChannelFirstd(keys=MODALIDADES),
    ConcatItemsd(keys=MODALIDADES, name='image', dim=0),
    NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
])

_modelo_superv = None


def cargar_modelo_supervivencia():
    """Carga el .joblib una sola vez (cache) y avisa si no existe."""
    global _modelo_superv
    if _modelo_superv is None:
        if not os.path.exists(MODELO_SUPERVIVENCIA):
            raise gr.Error(f"No se encontró el modelo de supervivencia: {MODELO_SUPERVIVENCIA}")
        _modelo_superv = joblib.load(MODELO_SUPERVIVENCIA)
    return _modelo_superv


# -----------------------------------------------------------------------------
# Segmentación
# -----------------------------------------------------------------------------
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

    faltantes = [m for m in MODALIDADES if m not in data_dict]
    if faltantes:
        raise gr.Error(f'Faltan modalidades: {faltantes}')

    tensor = tf(data_dict)['image'].unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = inferidor(inputs=tensor, network=model)
        resultado = (torch.sigmoid(logits) > 0.5).float()

    resultado_np = resultado.detach().cpu().numpy()
    fondo_2d = tensor[0, 1].cpu().numpy()
    vol_cerebro = nib.load(ruta_t1ce).get_fdata().astype(np.float32)
    return resultado_np, fondo_2d, vol_cerebro


# -----------------------------------------------------------------------------
# Métricas cuantitativas — única fuente de verdad (tabla + features del modelo)
# -----------------------------------------------------------------------------
def cuadrante_dominante(mascara: np.ndarray) -> int:
    """Divide el volumen en 8 octantes y devuelve (1-8) el de mayor volumen (0 si vacío)."""
    cx, cy, cz = (d // 2 for d in mascara.shape)
    octantes = [
        mascara[:cx, :cy, cz:], mascara[cx:, :cy, cz:],
        mascara[:cx, :cy, :cz], mascara[cx:, :cy, :cz],
        mascara[:cx, cy:, cz:], mascara[cx:, cy:, cz:],
        mascara[:cx, cy:, :cz], mascara[cx:, cy:, :cz],
    ]
    conteos = [int((o > 0.5).sum()) for o in octantes]
    return int(np.argmax(conteos)) + 1 if max(conteos) > 0 else 0


def distancia_al_centro(mascara: np.ndarray) -> float:
    """Distancia euclidiana del centro del volumen al centroide de la máscara."""
    coords = np.argwhere(mascara > 0.5)
    if coords.size == 0:
        return 0.0
    centro = np.array(mascara.shape) / 2
    return float(np.linalg.norm(coords.mean(axis=0) - centro))


def calcular_metricas(resultado_np: np.ndarray, vol_cerebro: np.ndarray) -> dict:
    tc, wt, et = resultado_np[0, 0], resultado_np[0, 1], resultado_np[0, 2]

    # Regiones exclusivas (misma lógica que la visualización 3D):
    necrotico = np.clip(tc - et, 0, 1)   # NCR/NET : núcleo sin parte activa
    edema = np.clip(wt - tc, 0, 1)       # ED      : tumor total sin núcleo

    wt_vol = int(wt.sum())
    brain_vol = int((vol_cerebro > 0).sum())
    healthy_vol = brain_vol - wt_vol

    return {
        "TC_volume": int(tc.sum()),
        "WT_volume": wt_vol,
        "ET_volume": int(et.sum()),
        "NCR/NET_volume": int(necrotico.sum()),
        "ED_volume": int(edema.sum()),
        "brain_vol": brain_vol,
        "healthy_br_vol": healthy_vol,
        "tumor_percen": wt_vol / brain_vol * 100 if brain_vol else 0.0,
        "healthy_br_percen": healthy_vol / brain_vol * 100 if brain_vol else 0.0,
        "WT_quadrant": cuadrante_dominante(wt),
        "TC_quadrant": cuadrante_dominante(tc),
        "Distance_mid_brain-tumor": distancia_al_centro(tc),
    }


def hacer_tabla(metricas: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "Región": ["TC — Tumor Core", "WT — Whole Tumor", "ET — Enhancing",
                   "Cerebro total", "Cerebro sano"],
        "Color 3D": ["Rojo", "Verde", "Amarillo", "—", "—"],
        "Vóxeles": [metricas["TC_volume"], metricas["WT_volume"], metricas["ET_volume"],
                    metricas["brain_vol"], metricas["healthy_br_vol"]],
    })


# -----------------------------------------------------------------------------
# Estimación de supervivencia
# -----------------------------------------------------------------------------
def clasificar_supervivencia(dias: float) -> str:
    if dias <= 200:
        return "Corta (Short-survivor)"
    if dias <= 600:
        return "Media (Mid-survivor)"
    return "Larga (Long-survivor)"


def construir_features(metricas: dict, edad: float) -> pd.DataFrame:
    """Selecciona y ordena las 12 variables predictoras del modelo."""
    valores = {**metricas, "Age": float(edad)}
    return pd.DataFrame([[valores[c] for c in FEATURES_MODELO]], columns=FEATURES_MODELO)


def estimar_supervivencia(X: pd.DataFrame) -> tuple[float, str]:
    modelo = cargar_modelo_supervivencia()
    orden = getattr(modelo, "feature_names_in_", None)
    if orden is not None:
        X = X[list(orden)]  # alinea el orden de columnas al que vio el modelo al entrenar
    dias = float(modelo.predict(X)[0])
    return dias, clasificar_supervivencia(dias)


# -----------------------------------------------------------------------------
# Visualización 2D
# -----------------------------------------------------------------------------
def actualizar_2d(resultado_np, fondo_2d, s1, s2, s3):
    if resultado_np is None:
        return None, None, None

    vol = resultado_np[0]
    s1, s2, s3 = int(s1), int(s2), int(s3)
    s1 = min(s1, vol.shape[1] - 1)
    s2 = min(s2, vol.shape[2] - 1)
    s3 = min(s3, vol.shape[3] - 1)

    def hacer_corte(plano_tumor, corte_cerebro):
        vmin = np.percentile(corte_cerebro, 1)
        vmax = np.percentile(corte_cerebro, 99)
        gris = np.clip((corte_cerebro - vmin) / (vmax - vmin + 1e-8), 0, 1)
        rgb = np.stack([gris, gris, gris], axis=-1)

        edema = (plano_tumor[1] - plano_tumor[0]) > 0.5
        tc = plano_tumor[0] > 0.5
        et = plano_tumor[2] > 0.5

        rgb[edema] = [0.2, 0.5, 1.0]
        rgb[tc] = [1.0, 0.3, 0.1]
        rgb[et] = [1.0, 1.0, 0.0]
        return rgb.astype(np.float32)

    return (
        hacer_corte(vol[:, s1, :, :], fondo_2d[s1, :, :]),
        hacer_corte(vol[:, :, s2, :], fondo_2d[:, s2, :]),
        hacer_corte(vol[:, :, :, s3], fondo_2d[:, :, s3]),
    )


# -----------------------------------------------------------------------------
# Visualización 3D
# -----------------------------------------------------------------------------
def actualizar_3d(resultado_np, vol_cerebro, show_tc, show_wt, show_et, show_brain):
    if resultado_np is None:
        return None

    mascara_tc = resultado_np[0, 0]
    mascara_wt = resultado_np[0, 1]
    mascara_et = resultado_np[0, 2]

    edema_puro = np.clip(mascara_wt - mascara_tc, 0, 1)
    tc_puro = np.clip(mascara_tc - mascara_et, 0, 1)

    capas = [
        ('cerebro', vol_cerebro,
         [200, 200, 200, 25], show_brain,
         float(np.percentile(vol_cerebro[vol_cerebro > 0], 30))),
        ('Edema', edema_puro, [50, 200, 50, 90], show_wt, 0.5),
        ('TC', tc_puro, [220, 50, 50, 180], show_tc, 0.5),
        ('ET', mascara_et, [255, 230, 0, 255], show_et, 0.5),
    ]

    piezas = []
    for nombre, arr, color, activo, umbral in capas:
        if not activo:
            continue

        a = np.ascontiguousarray(arr.astype(np.float32))
        if a.max() < umbral:
            continue

        malla = Volume(a).isosurface(value=umbral)
        if nombre == "cerebro":
            malla = malla.decimate(fraction=0.01)
        if malla.npoints == 0:
            continue

        verts = np.array(malla.vertices)
        caras = np.array(malla.cells)
        tm = trimesh.Trimesh(vertices=verts, faces=caras, process=False)
        tm.visual.vertex_colors = np.tile(np.array(color, dtype=np.uint8), (len(verts), 1))
        piezas.append(tm)

    if not piezas:
        return None

    ruta_glb = os.path.abspath("tumor.glb")
    trimesh.Scene(piezas).export(ruta_glb)
    return ruta_glb


# -----------------------------------------------------------------------------
# Interfaz
# -----------------------------------------------------------------------------
with gr.Blocks(title="Segmentador BraTS") as app:
    state_resultado = gr.State(value=None)
    state_fondo = gr.State(value=None)
    state_cerebro = gr.State(value=None)

    gr.Markdown("# Segmentador de tumores cerebrales — BraTS 2018")
    gr.Markdown("**1:** subir las 4 modalidades (t1, t1ce, t2, flair). **2:** *predecir*. **3:** configurar.")

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
            sl_axi = gr.Slider(0, 154, value=77, step=1, label="Axial (Z)")

            gr.Markdown("### Capas 3D")
            cb_tc = gr.Checkbox(value=True, label="TC — Tumor Core (rojo)")
            cb_wt = gr.Checkbox(value=True, label="WT — Whole Tumor (verde)")
            cb_et = gr.Checkbox(value=True, label="ET — Enhancing (amarillo)")
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
        metricas = calcular_metricas(resultado_np, vol_cerebro)
        sag, cor, axi = actualizar_2d(resultado_np, fondo_2d, s1, s2, s3)
        glb = actualizar_3d(resultado_np, vol_cerebro, tc, wt, et, brain)
        return resultado_np, fondo_2d, vol_cerebro, sag, cor, axi, glb, hacer_tabla(metricas)

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

    # --- Estimación de supervivencia (al final de la página) --------------------
    gr.Markdown("## Estimación de supervivencia")
    edad = gr.Number(label="Edad del paciente (años)", value=60, precision=0)
    btn_superv = gr.Button("Predecir supervivencia", variant="primary")
    salida_superv = gr.Markdown()

    def on_supervivencia(resultado_np, vol_cerebro, edad):
        if resultado_np is None:
            raise gr.Error("Primero ejecuta la segmentación con el botón 'Predecir'.")
        X = construir_features(calcular_metricas(resultado_np, vol_cerebro), edad)
        dias, grupo = estimar_supervivencia(X)
        return (
            f"### Resultado\n"
            f"- **{dias:.0f} días** (~{dias / 30:.1f} meses)\n"
            f"- Grupo pronóstico: **{grupo}**"
        )

    btn_superv.click(
        fn=on_supervivencia,
        inputs=[state_resultado, state_cerebro, edad],
        outputs=[salida_superv],
    )

app.launch()