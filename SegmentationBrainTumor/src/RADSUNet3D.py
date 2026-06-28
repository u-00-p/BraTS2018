"""IMPORTS DE LA RED NEURONAL RESIDUAL ATTENTION U NET 3D"""
import torch
import torch.nn as nn
import torch.nn.functional as F

"""CLASE QUE DEFINE EL COMPORTAMIENTO RESIDUAL DE LAS CONVOLUCIONES"""
class ResDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResDoubleConv, self).__init__()
        self.main_path = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels)
        )
        if in_channels != out_channels:

            # Camino corto
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.InstanceNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x_main = self.main_path(x)
        x_shortcut = self.shortcut(x)
        out = x_main + x_shortcut
        return self.relu(out)

"""CLASE QUE DEFINE EL COMPORTAMIENTO DE ATENCION DE LAS CONVOLUCIONES"""
class AttentionBlock3d(nn.Module):
    def __init__(self,F_g, F_l,F_int):
        super(AttentionBlock3d, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm3d(1, affine=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi_out = self.relu(g1+x1)
        attention_map = self.psi(psi_out)
        return x * attention_map

"""CLASE PRINCIPAL DE LA RED NEURONAL"""
class RadsUNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, features=[16,32,64,128], deep_supervision=True):
        super(RadsUNet3D, self).__init__()
        self.deep_supervision = deep_supervision
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        for feature in features:
            self.downs.append(ResDoubleConv(in_channels, feature))
            in_channels=feature
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose3d(feature*2, feature, kernel_size=2, stride=2))
            self.attentions.append(AttentionBlock3d(F_g=feature, F_l=feature, F_int=feature//2))
            self.ups.append(ResDoubleConv(feature*2, feature))
        self.bottleneck = ResDoubleConv(features[-1], features[-1]*2)
        self.dropout = nn.Dropout3d(p=0.2)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        if deep_supervision:
            self.ds_heads = nn.ModuleList([
                nn.Conv3d(f, out_channels, kernel_size=1) for f in list(reversed(features))[:-1]
            ])
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x); skip_connections.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        x = self.dropout(x)
        skip_connections = skip_connections[::-1]

        ds_outputs = []
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='trilinear', align_corners=False)
            skip_filtered = self.attentions[idx//2](g=x, x=skip_connection)
            x = self.ups[idx+1](torch.cat((skip_filtered, x), dim=1))

            level = idx // 2
            if self.deep_supervision and self.training and level < len(self.ds_heads):
                ds_outputs.append(self.ds_heads[level](x))

        main_out = self.final_conv(x)
        if self.deep_supervision and self.training:
            return [main_out] + ds_outputs[::-1]
        return main_out
    
"""PRUEBA DE EJECUCION"""
if __name__ == '__main__':
    x = torch.rand((1, 4, 128, 128, 128))
    print(f'{x.shape}')
    model = RadsUNet3D(in_channels=4, out_channels=3, features=[16,32,64,128])
    model.eval()
    with torch.no_grad():
        predict = model(x)
    print(f'{predict.shape}')