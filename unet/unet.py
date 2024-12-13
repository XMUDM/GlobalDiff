# _*_ encoding: utf-8 _*_
# @ Author  : lmx
# @ Time : 2024/5/7
# @ desc : unet module

import torch.nn as nn
import torch.nn.functional as F
import torch
class UpConvSample(nn.Module):
    """
        转置卷积上采样,上采样,特征图大小扩大2倍,通道数减半
    """
    def __init__(self, in_channels):
        super(UpConvSample, self).__init__()
        self.up=nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2),
                nn.LeakyReLU(),
            )
    def forward(self, x):
        x = self.up(x)
        return x
    
class UpInterplateSample(nn.Module):
    """
        双线性插值上采样，上采样,特征图大小扩大2倍,通道数减半
    """
    def __init__(self, in_channels):
        super(UpInterplateSample, self).__init__()
    
        self.up=nn.Conv2d(in_channels, in_channels // 2, 1, 1)
    
    def forward(self, x,featuremap):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.up(x)
        return torch.cat([x,featuremap],dim=1)

class DownConvSample(nn.Module):
    """
        卷积下采样,下采样,特征图大小缩小2倍,通道数不变

    """
    def __init__(self,in_channels):
        super(DownConvSample, self).__init__()
        self.down=nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,stride=2,padding=1),  # 卷积操作
            nn.LeakyReLU()
        )
    def forward(self, x):
        x = self.down(x)
        return x

class ConvBlock(nn.Module):
    """
        卷积块,包含卷积,BN,LeakyReLU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,padding_mode='reflect'),  # 卷积操作
            nn.BatchNorm2d(out_channels),  # 批量归一化
            # 防止过拟合
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,padding_mode='reflect'),  # 再次卷积
            nn.BatchNorm2d(out_channels),  # 再次批量归一化
            # 防止过拟合
            nn.Dropout(0.4),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.layer(x)
        return x
    
# class  Unet(nn.Module):
#     """
#         U-Net模型
#     """

#     def __init__(self):
#         super(Unet, self).__init__()

#         self.c1=ConvBlock(1,32)
#         self.d1=DownConvSample(32)
#         self.c2=ConvBlock(32,64)
#         self.d2=DownConvSample(64)
#         self.c3=ConvBlock(64,128)
#         self.u1=UpInterplateSample(128)
#         self.c4=ConvBlock(128,64)
#         self.u2=UpInterplateSample(64)
#         self.c5=ConvBlock(64,32)

#         self.pred = torch.nn.Conv2d(32, 1, 3, 1, 1)
#         self.sig = torch.nn.Sigmoid()

    
#     def forward(self, x):
#         # 编码器
#         r1=self.c1(x)
#         r2=self.c2(self.d1(r1))
#         r3=self.c3(self.d2(r2))
#         # 解码器
#         o1=self.c4(self.u1(r3,r2))
#         o2=self.c5(self.u2(o1,r1))

#         out=self.sig(self.pred(o2))
#         return  out

class  Unet(nn.Module):
    """
        U-Net模型
    """

    def __init__(self):
        super(Unet, self).__init__()

        self.c1=ConvBlock(1,32)
        self.d1=DownConvSample(32)
        self.c2=ConvBlock(32,64)
        self.d2=DownConvSample(64)
        self.c3=ConvBlock(64,128)
        self.u1=UpInterplateSample(128)
        self.c4=ConvBlock(128,64)
        self.u2=UpInterplateSample(64)
        self.c5=ConvBlock(64,32)

        self.pred = torch.nn.Conv2d(32, 1, 3, 1, 1)
        self.sig = torch.nn.Sigmoid()

    
    def forward(self, x):
        # 编码器
        r1=self.c1(x)
        r2=self.c2(self.d1(r1))
        # 解码器
        o2=self.c5(self.u2(r2,r1))
        out=self.sig(self.pred(o2))
        return  out
