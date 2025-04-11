# 层定义与模型定义(Unet)
# 输入与输出大小相同,目前固定为128*128

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchaudio
import datasets
import numpy
import config



# 时间编码层,默认编码维度32
class TimeEmbedding(nn.Module):
    # 时间编码序列长度，输入时间步（1~几千），输出默认32维的向量
    def __init__(self,dim):
        super().__init__()
        self.dim=dim

    def forward(self,t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# 上采样层
class UpSample(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        # 这样设置参数可以使输出图像的宽高乘以二
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.up(x)


# 下采样层
class DownSample(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.down=nn.Sequential(
            nn.Conv2d(in_ch,out_ch,4,2,1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.down(x)


# 主模型
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64,64,3,padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64)
        )

        self.down1=DownSample(64,64)
        self.conv2=nn.Sequential(
            nn.Conv2d(64,128,3,padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128,128,3,padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128)
        )
        self.down2 = DownSample(128, 128)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.down3 = DownSample(256, 256)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.down4 = DownSample(512, 512)
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024)
        )
        self.up1 = UpSample(1024,512)
        # conv6-9均具有残差输入
        self.conv6=nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.up2 = UpSample(512, 256)
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.up3 = UpSample(256, 128)
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128)
        )
        self.up4 = UpSample(128, 64)
        self.conv9 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 3, 1, padding=0)

        )
        self.time_embedder=TimeEmbedding(32)
        self.time_linear1 = nn.Sequential(
            # nn.Linear(32,32),
            # nn.GELU(),
            nn.Linear(32,1024),
            nn.GELU(),

        )
        self.time_linear2 = nn.Sequential(
            # nn.Linear(32,32),
            # nn.GELU(),
            nn.Linear(32,512),
            nn.GELU(),

        )
        self.time_linear3 = nn.Sequential(
            # nn.Linear(32,32),
            # nn.GELU(),
            nn.Linear(32,256),
            nn.GELU(),

        )
        self.time_linear4 = nn.Sequential(
            # nn.Linear(32,32),
            # nn.GELU(),
            nn.Linear(32,128),
            nn.GELU(),

        )




    def forward(self,x,t):
        t_embedded=self.time_embedder(t)
        res1=self.conv1(x)
        x=self.down1(res1)
        res2=self.conv2(x)
        x=self.down2(res2)
        res3=self.conv3(x)
        x=self.down3(res3)
        res4 = self.conv4(x)
        x=self.down4(res4)
        x = self.conv5(x)

        x=self.up1(x)
        x=torch.cat((x,res4),dim=1)
        x=x+self.time_linear1(t_embedded)[(...,) + (None,) * 2]#加这堆奇怪的符号是为了补全维度，才能不报错
        x=self.conv6(x)

        x = self.up2(x)
        x = torch.cat((x, res3), dim=1)
        x = x + self.time_linear2(t_embedded)[(...,) + (None,) * 2]
        x = self.conv7(x)

        x = self.up3(x)
        x = torch.cat((x, res2), dim=1)
        x = x + self.time_linear3(t_embedded)[(...,) + (None,) * 2]
        x = self.conv8(x)

        x = self.up4(x)
        x = torch.cat((x, res1), dim=1)
        x = x + self.time_linear4(t_embedded)[(...,) + (None,) * 2]
        x = self.conv9(x)

        return x










device=config.device
net=Network().to(device)
print("Num params: ", sum(p.numel() for p in net.parameters()))