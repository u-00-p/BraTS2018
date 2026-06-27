"""IMPORTACIONES PARA EL ENTRENAMIENTO"""
import os
from glob import glob
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference

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
            LoadImaged(keys=['image', 'label']),
            EnsureChannelFirstd(keys='image'),
            ConvertToMultiChannelBasedOnBratsClassesd(keys='label'),
            #Spacingd(keys=['image', 'label'], pixdim=(1,1,1)),
            NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
            CropForegroundd(keys=['image', 'label'], source_key='image'),
            #Resized(keys=['image', 'label'], spatial_size=[128,128,128], mode=('trilinear', 'nearest')),
            SpatialPadd(keys=['image', 'label'], spatial_size=[128,128,128]),
            RandCropByPosNegLabeld(keys=['image', 'label'], label_key='label',spatial_size=[128,128,128], pos=1, neg=1, num_samples=1,),
            #RandSpatialCropd(keys=['image', 'label'], roi_size=[128,128,128], random_size=False),
            RandRotate90d(keys=['image', 'label'], prob=0.5, spatial_axes=(0, 1)),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
            RandGaussianNoised(keys=['image'], prob=0.2, mean=0.0, std=0.1),
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

"""SE DEFINEN LOS LOADER DE LOS ARCHIVOS DE ENTRENAMIENTO Y VALIDACION"""
def lotes(train_files, val_files, train_transforms, val_transforms):
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, pin_memory=True)

    return train_loader, val_loader

"""SE DEFINE LA LOGICA DE ENTRENAMIENTO DEL MODELO"""
def train_model(train_loader, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Dispositivo de entrenamiento {device}')
    use_amp = (device.type == 'cuda')

    model = RadsUNet3D(in_channels=4, out_channels=3, features=[32,64,128,256]).to(device)
    dice_loss = DiceLoss(include_background=True, to_onehot_y=False, sigmoid=True, squared_pred=True)
    bce_loss  = nn.BCEWithLogitsLoss()
    def criterion(pred, target):
        return dice_loss(pred, target.float()) + bce_loss(pred, target.float())
    optimizer = optim.AdamW(model.parameters(),lr=3e-4, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    num_epocas = 450
    mejor_loss_validacion = float('inf')
    save_dir = 'model_raunet3d.pth'
    paciencia = 25
    epocas_sin_mejora = 0

    pasos_acumulacion = 8
    
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
        
        print(f'Loss promedio del entrenamiento: {loss_entrenamiento_acumulada/len(train_loader)}')

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
                print(f"Validando {idx_val + 1} / {len(val_loader)}")
        
        loss_promedio_val = loss_validacion_acumulada / len(val_loader)
        scheduler.step(loss_promedio_val)
        print(f"LR actual: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"Loss promedio validacion: {loss_promedio_val:.4f}")

        if loss_promedio_val < mejor_loss_validacion:
            print(f"La perdida bajo de {mejor_loss_validacion:.4f} a {loss_promedio_val:.4f}")
            mejor_loss_validacion = loss_promedio_val
            torch.save(model.state_dict(), save_dir)
            epocas_sin_mejora = 0
            print("Guardado exitosamente.")
        else:
          epocas_sin_mejora += 1
          if epocas_sin_mejora >= paciencia:
              print(f"Early stopping: {paciencia} epocas sin mejorar.")
              break

"""EJECUCION"""
if __name__ == '__main__':
    train_files, val_files = definir_rutas_diccionarios()
    train_transforms, val_transforms = transformaciones()
    train_loader, val_loader = lotes(train_files, val_files, train_transforms, val_transforms)
    train_model(train_loader, val_loader)
