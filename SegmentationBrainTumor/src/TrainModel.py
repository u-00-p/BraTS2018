"""IMPORTACIONES PARA EL ENTRENAMIENTO"""
import os
from glob import glob
import random
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from RADSUNet3D import RadsUNet3D

from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    EnsureChannelFirstd,
    #Spacingd
    NormalizeIntensityd,
    CropForegroundd,
    #Resized,
    ConvertToMultiChannelBasedOnBratsClassesd,
    RandRotate90d,
    RandFlipd,
    RandGaussianNoised,
    MapTransform,
    #RandSpatialCropd,
    RandCropByPosNegLabeld, 
    SpatialPadd
)
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

"""SEMILLA GLOBAL DE REPRODUCIBILIDAD"""
SEED = 42

"""CLASE QUE DESACTIVA DE MANERA RANDOM 1 O 2 MODALIDADES DE IMAGEN"""
class RandModalityDropoutd(MapTransform):
    def __init__(self,keys,prob=0.5):
        super().__init__(keys)
        self.prob = prob

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if random.random() < self.prob:
                num_drop = random.choice([1,2])
                canales_apagar = random.sample(range(4), num_drop)
                for c in canales_apagar:
                    d[key][c,...]=0.0
        return d
    
"""SE DEFINEN LOS DICCIONARIOS CORRESPONDIENTES CON LAS IMAGENES"""
def definir_rutas_diccionarios():
    data = os.getcwd()
    data = os.path.dirname(data)
    data_split = os.path.join(data,'data_split')
    os.makedirs(data_split, exist_ok=True)
    data = os.path.join(data, 'data')
    data_split_train = os.path.join(data_split, 'train')
    data_split_val = os.path.join(data_split, 'val')

    mask_train = os.path.join(data_split_train,'masks')
    mask_val = os.path.join(data_split_val,'masks')
    images_train = os.path.join(data_split_train,'images')
    images_val = os.path.join(data_split_val,'images')

    flair_train = sorted(glob(os.path.join(images_train, '*_flair.nii.gz')))
    t1ce_train  = sorted(glob(os.path.join(images_train, '*_t1ce.nii.gz')))
    t1_train    = sorted(glob(os.path.join(images_train, '*_t1.nii.gz')))
    t2_train    = sorted(glob(os.path.join(images_train, '*_t2.nii.gz')))
    seg_train   = sorted(glob(os.path.join(mask_train,   '*_seg.nii.gz')))

    train_files = [
        {'image': [f, c, o, d], 'label': s}
        for f, c, o, d, s in zip(flair_train, t1ce_train, t1_train, t2_train, seg_train)
    ]

    flair_val = sorted(glob(os.path.join(images_val, '*_flair.nii.gz')))
    t1ce_val  = sorted(glob(os.path.join(images_val, '*_t1ce.nii.gz')))
    t1_val    = sorted(glob(os.path.join(images_val, '*_t1.nii.gz')))
    t2_val    = sorted(glob(os.path.join(images_val, '*_t2.nii.gz')))
    seg_val   = sorted(glob(os.path.join(mask_val,   '*_seg.nii.gz')))

    val_files = [
        {'image': [f, c, o, d], 'label': s}
        for f, c, o, d, s in zip(flair_val, t1ce_val, t1_val, t2_val, seg_val)
    ]

    print('train:', len(train_files), ' val:', len(val_files))
    if len(train_files)==0 or len(val_files)==0:
        print("Ruta equivocada!")
        raise RuntimeError("No ahy archivos")

    return train_files, val_files

"""SE DEFINEN LAS TRANSFORMACIONES NECESARIAS A LAS IMAGENES"""
def transformaciones():
    train_transforms = Compose(
        [
            LoadImaged(keys=['image', 'label']), #Carga las imagenes y apila las 4 modalidades
            EnsureChannelFirstd(keys='image'), # Garantiza el formato con los canales primero
            ConvertToMultiChannelBasedOnBratsClassesd(keys='label'), #Convierte la mascara en 3 canales correspondientes
            #Spacingd(keys=['image', 'label'], pixdim=(1,1,1)),
            NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True), #Normaliza por canal e ignora el fondo=0
            CropForegroundd(keys=['image', 'label'], source_key='image'), #Recorta la parte negra 
            #Resized(keys=['image', 'label'], spatial_size=[128,128,128], mode=('trilinear', 'nearest')),
            SpatialPadd(keys=['image', 'label'], spatial_size=[128,128,128]), #Rellena a 128^3 los que sean mas chicos 
            RandCropByPosNegLabeld(keys=['image', 'label'], label_key='label',spatial_size=[128,128,128], pos=1, neg=1, num_samples=1,), #Recorta un cubo aleatorio y balance el fondo con el tumor
            #RandSpatialCropd(keys=['image', 'label'], roi_size=[128,128,128], random_size=False),
            
            #Data augmention
            RandRotate90d(keys=['image', 'label'], prob=0.5, spatial_axes=(0, 1)), #Rota aleatoriamente 90grados
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0), #Efecto espejo
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
            RandGaussianNoised(keys=['image'], prob=0.2, mean=0.0, std=0.1), #Ruido
            #RandModalityDropoutd(keys=['image'], prob=0.5),
            ToTensord(keys=['image', 'label'])
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=['image', 'label']),
            EnsureChannelFirstd(keys='image'),
            ConvertToMultiChannelBasedOnBratsClassesd(keys='label'),
            #Spacingd(keys=['image', 'label'], pixdim=(1,1,1), mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
            CropForegroundd(keys=['image', 'label'], source_key='image'),
            #Resized(keys=['image', 'label'], spatial_size=[128,128,128], mode=('trilinear', 'nearest')),
            ToTensord(keys=['image', 'label'])
        ]
    )

    return train_transforms, val_transforms

"""SE DEFINE LA FUNCION PARA SABER QUE CRITERIO DE PESOS DAR"""
def deep_supervision_loss(outputs, target, base_criterion):

    '''Se predicen las 4 capas del encoder y decoder, dandole diferentes pesos a cada una despues de esto se pondera el error y la red aprende'''
    '''Se ponderan las predicciones en las 4 resoluciones diferentes, dandole mas peso a la primera'''
    target = target.float()
    pesos = [1, 0.5, 0.25, 0.125][:len(outputs)]
    pesos = [p / sum(pesos) for p in pesos]
    total = 0.0
    for salida, peso in zip(outputs, pesos):
        if salida.shape[2:] != target.shape[2:]:
            t = F.interpolate(target, size=salida.shape[2:], mode='nearest')
        else:
            t = target

        total = total + peso * base_criterion(salida, t)
    return total

"""SE GUARDA EL HISTORIAL DE ENTRENAMIENTO EN UN TXT Y SE GRAFICA LA CURVA DE PERDIDA Y DE DICE"""
def guardar_historial_y_grafica(historial, hiperparametros, mejor_epoca, mejor_dice_validacion,
                                 save_dir, hora_inicio, log_path='historial_entrenamiento.txt',
                                 grafica_path='curva_entrenamiento.png'):
    hora_fin = datetime.now()

    epocas       = historial['epoca']
    train_losses = historial['train_loss']
    val_losses   = historial['val_loss']
    dice_tc      = historial['dice_tc']
    dice_wt      = historial['dice_wt']
    dice_et      = historial['dice_et']
    dice_medio   = historial['dice_medio']
    lrs          = historial['lr']
    guardados    = historial['guardado']

    if mejor_epoca is not None and mejor_epoca in epocas:
        idx_mejor = epocas.index(mejor_epoca)
        dice_medio_mejor = dice_medio[idx_mejor]
        val_loss_mejor   = val_losses[idx_mejor]
    else:
        dice_medio_mejor = None
        val_loss_mejor   = None

    """SE ESCRIBE EL TXT CON TODA LA INFORMACION DEL ENTRENAMIENTO PARA NO TENER QUE REENTRENAR"""
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=== HISTORIAL DE ENTRENAMIENTO ===\n")
        f.write(f"Fecha inicio: {hora_inicio.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Fecha fin: {hora_fin.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duracion total: {hora_fin - hora_inicio}\n")
        f.write(f"Epocas ejecutadas: {len(epocas)} / {hiperparametros['num_epocas']}\n")
        f.write(f"Mejor epoca (modelo guardado, criterio = Dice mas alto): {mejor_epoca}\n")
        f.write(f"Mejor Dice medio de validacion: {mejor_dice_validacion:.6f}\n")
        if val_loss_mejor is not None:
            f.write(f"Loss de validacion en esa epoca: {val_loss_mejor:.6f}\n")
        f.write(f"Modelo guardado en: {save_dir}\n\n")

        f.write("Hiperparametros:\n")
        for clave, valor in hiperparametros.items():
            f.write(f"  {clave}: {valor}\n")
        f.write("\n")

        cabecera = (f"{'Epoca':>6} | {'Loss train':>11} | {'Loss val':>10} | "
                    f"{'Dice TC':>8} | {'Dice WT':>8} | {'Dice ET':>8} | {'Dice med':>8} | {'LR':>10} | Guardado")
        f.write(cabecera + "\n")
        for e, tl, vl, dtc, dwt, det, dm, lr, g in zip(
                epocas, train_losses, val_losses, dice_tc, dice_wt, dice_et, dice_medio, lrs, guardados):
            marca = "SI" if g else ""
            f.write(f"{e:>6} | {tl:>11.6f} | {vl:>10.6f} | {dtc:>8.4f} | {dwt:>8.4f} | "
                    f"{det:>8.4f} | {dm:>8.4f} | {lr:>10.2e} | {marca}\n")

    print(f"Historial guardado en {log_path}")

    """SE GUARDAN LOS DATOS CRUDOS EN CSV PARA REGENERAR FIGURAS DEL PAPER SIN REENTRENAR"""
    csv_path = 'historial_entrenamiento.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("epoca,train_loss,val_loss,dice_tc,dice_wt,dice_et,dice_medio,lr,guardado\n")
        for e, tl, vl, dtc, dwt, det, dm, lr, g in zip(
                epocas, train_losses, val_losses, dice_tc, dice_wt, dice_et, dice_medio, lrs, guardados):
            f.write(f"{e},{tl},{vl},{dtc},{dwt},{det},{dm},{lr},{int(g)}\n")
    print(f"Datos crudos guardados en {csv_path}")

    """SE GRAFICAN 2 PANELES: ARRIBA LA PERDIDA, ABAJO EL DICE (AMBOS COMPARTEN EL EJE DE EPOCAS)"""
    color_train = '#0072B2'
    color_val = '#E69F00'
    color_guardado = '#009E73'
    color_tc = '#4C72B0'
    color_wt = '#55A868'
    color_et = '#C44E52'

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    ax1.plot(epocas, train_losses, label='Loss entrenamiento', color=color_train, linewidth=1.8)
    ax1.plot(epocas, val_losses,   label='Loss validacion',    color=color_val,   linewidth=1.8)
    if mejor_epoca is not None and val_loss_mejor is not None:
        ax1.axvline(mejor_epoca, color='#999999', linestyle='--', linewidth=1)
        ax1.scatter([mejor_epoca], [val_loss_mejor], color=color_guardado, s=110, zorder=5,
                    marker='*', label='Modelo guardado (mejor Dice)')
    ax1.set_ylabel('Loss')
    ax1.set_title('Curva de entrenamiento')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, color='#dddddd', linewidth=0.6)
    ax1.legend(frameon=False)

    ax2.plot(epocas, dice_tc,    label='Dice TC', color=color_tc, linewidth=1.5)
    ax2.plot(epocas, dice_wt,    label='Dice WT', color=color_wt, linewidth=1.5)
    ax2.plot(epocas, dice_et,    label='Dice ET', color=color_et, linewidth=1.5)
    ax2.plot(epocas, dice_medio, label='Dice medio', color='#333333', linewidth=2.2, linestyle='--')
    if mejor_epoca is not None:
        ax2.axvline(mejor_epoca, color='#999999', linestyle='--', linewidth=1)
        if dice_medio_mejor is not None:
            ax2.scatter([mejor_epoca], [dice_medio_mejor], color=color_guardado, s=110, zorder=5, marker='*')
            ax2.annotate(
                f"Modelo guardado\nEpoca {mejor_epoca}\nDice medio: {dice_medio_mejor:.4f}",
                xy=(mejor_epoca, dice_medio_mejor),
                xytext=(15, -35), textcoords='offset points',
                fontsize=9, color=color_guardado,
                arrowprops=dict(arrowstyle='->', color=color_guardado, lw=1)
            )
    ax2.set_xlabel('Epoca')
    ax2.set_ylabel('Dice')
    ax2.set_ylim(0, 1.02)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, color='#dddddd', linewidth=0.6)
    ax2.legend(frameon=False, ncol=2)

    plt.tight_layout()
    plt.savefig(grafica_path, dpi=150)
    plt.close(fig)

    print(f"Curva de entrenamiento guardada en {grafica_path}")

"""SE DEFINEN LOS LOADER DE LOS ARCHIVOS DE ENTRENAMIENTO Y VALIDACION"""
def lotes(train_files, val_files, train_transforms, val_transforms):
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, pin_memory=True,
                            persistent_workers=True, prefetch_factor=2)

    return train_loader, val_loader

"""SE DEFINE LA LOGICA DE ENTRENAMIENTO DEL MODELO"""
def train_model(train_loader, val_loader):
    set_determinism(seed=SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Dispositivo de entrenamiento {device}')
    use_amp = (device.type == 'cuda')

    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = RadsUNet3D(in_channels=4, out_channels=3, features=[32,64,128,256]).to(device)
    dice_loss = DiceLoss(include_background=True, to_onehot_y=False, sigmoid=True, squared_pred=True)
    bce_loss  = nn.BCEWithLogitsLoss()
    def criterion(pred, target):
        return dice_loss(pred, target.float()) + bce_loss(pred, target.float())
    optimizer = optim.AdamW(model.parameters(),lr=3e-4, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    dice_metric = DiceMetric(include_background=True, reduction='mean_batch', ignore_empty=False)

    num_epocas = 450
    mejor_dice_validacion = -float('inf')
    mejor_epoca = None
    save_dir = 'model_brats.pth'
    paciencia = 25
    epocas_sin_mejora = 0

    pasos_acumulacion = 8

    """SE GUARDAN LOS HIPERPARAMETROS Y EL HISTORIAL PARA PODER CONSULTARLOS SIN REENTRENAR"""
    hiperparametros = {
        'arquitectura': 'RadsUNet3D',
        'features': [32, 64, 128, 256],
        'semilla': SEED,
        'learning_rate_inicial': 3e-4,
        'weight_decay': 1e-5,
        'pasos_acumulacion': pasos_acumulacion,
        'paciencia_early_stopping': paciencia,
        'num_epocas': num_epocas,
        'criterio_guardado': 'mejor Dice medio de validacion',
        'convencion_dice': 'BraTS (ignore_empty=False)',
        'dispositivo': str(device),
    }
    historial = {'epoca': [], 'train_loss': [], 'val_loss': [],
                 'dice_tc': [], 'dice_wt': [], 'dice_et': [], 'dice_medio': [],
                 'lr': [], 'guardado': []}
    hora_inicio = datetime.now()

    try:
        for epoch in range(num_epocas):
            print(f'Epoca {epoch+1} / {num_epocas}')
            model.train()
            loss_entrenamiento_acumulada = 0.0

            optimizer.zero_grad()

            for batch_idx, batch_data in enumerate(train_loader):
                imagenes = batch_data['image'].to(device)
                mascaras_reales = batch_data['label'].to(device)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    salidas = model(imagenes)
                    loss_real = deep_supervision_loss(salidas, mascaras_reales, criterion)

                    loss_normalizada = loss_real / pasos_acumulacion

                scaler.scale(loss_normalizada).backward()

                loss_entrenamiento_acumulada += loss_real.item()

                if ((batch_idx + 1) % pasos_acumulacion == 0) or (batch_idx + 1 == len(train_loader)):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                print(f'Batch {batch_idx+1} Loss:{loss_real.item():.4f}')

            loss_promedio_train = loss_entrenamiento_acumulada / len(train_loader)
            print(f'Loss promedio del entrenamiento: {loss_promedio_train}')

            model.eval()
            loss_validacion_acumulada = 0.0

            with torch.no_grad():
                for idx_val, batch_data in enumerate(val_loader):
                    imagenes_val = batch_data['image'].to(device)
                    mascaras_reales_val = batch_data['label'].to(device)

                    with torch.amp.autocast('cuda', enabled=use_amp):
                        predicciones_val = sliding_window_inference(
                            inputs=imagenes_val,
                            roi_size=(128,128,128),
                            sw_batch_size=1,
                            predictor=model,
                            overlap=0.5,
                        )
                        loss_val = criterion(predicciones_val, mascaras_reales_val)

                    loss_validacion_acumulada += loss_val.item()

                    pred_bin = (torch.sigmoid(predicciones_val.float()) > 0.5).float()
                    dice_metric(y_pred=pred_bin, y=mascaras_reales_val)

                    print(f"Validando {idx_val + 1} / {len(val_loader)}")

            loss_promedio_val = loss_validacion_acumulada / len(val_loader)

            dice_por_region = dice_metric.aggregate()
            dice_metric.reset() 
            dice_tc    = dice_por_region[0].item()
            dice_wt    = dice_por_region[1].item()
            dice_et    = dice_por_region[2].item()
            dice_medio = torch.nanmean(dice_por_region).item()

            scheduler.step(loss_promedio_val)
            print(f"LR actual: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"Loss promedio validacion: {loss_promedio_val:.4f}")
            print(f"Dice val -> TC={dice_tc:.4f} WT={dice_wt:.4f} ET={dice_et:.4f} | medio={dice_medio:.4f}")

            guardado_esta_epoca = False
            if dice_medio > mejor_dice_validacion:
                if mejor_dice_validacion == -float('inf'):
                    print(f"Primer modelo guardado (Dice medio: {dice_medio:.4f})")
                else:
                    print(f"El Dice medio subio de {mejor_dice_validacion:.4f} a {dice_medio:.4f}")
                mejor_dice_validacion = dice_medio
                mejor_epoca = epoch + 1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': getattr(model, '_orig_mod', model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dice_medio': mejor_dice_validacion,
                    'val_loss': loss_promedio_val,
                }, save_dir)
                epocas_sin_mejora = 0
                guardado_esta_epoca = True
                print("Guardado exitosamente.")
            else:
                epocas_sin_mejora += 1

            historial['epoca'].append(epoch + 1)
            historial['train_loss'].append(loss_promedio_train)
            historial['val_loss'].append(loss_promedio_val)
            historial['dice_tc'].append(dice_tc)
            historial['dice_wt'].append(dice_wt)
            historial['dice_et'].append(dice_et)
            historial['dice_medio'].append(dice_medio)
            historial['lr'].append(optimizer.param_groups[0]['lr'])
            historial['guardado'].append(guardado_esta_epoca)

            if epocas_sin_mejora >= paciencia:
                print(f"Early stopping: {paciencia} epocas sin mejorar el Dice.")
                break
    finally:
        """SE GUARDA EL HISTORIAL Y LA GRAFICA AUNQUE EL ENTRENAMIENTO SE INTERRUMPA (ERROR O CTRL+C)"""
        if len(historial['epoca']) > 0:
            guardar_historial_y_grafica(historial, hiperparametros, mejor_epoca, mejor_dice_validacion,
                                         save_dir, hora_inicio)

"""EJECUCION"""
if __name__ == '__main__':
    train_files, val_files = definir_rutas_diccionarios()
    train_transforms, val_transforms = transformaciones()
    train_loader, val_loader = lotes(train_files, val_files, train_transforms, val_transforms)
    train_model(train_loader, val_loader)