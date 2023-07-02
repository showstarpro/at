import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Based_AutoEncoder_More(nn.Module):
    def __init__(self):
        super(Based_AutoEncoder_More, self).__init__()
        self.encoder = Based_Encoder()
        self.decoder = Based_Decoder()
    
    def encode(self, x):
        z = self.encoder(x)
        return z
    def decode(self, z):
        y = self.decoder(z)
        return y

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        #x = x.unsqueeze(1)
        return x


class Based_Encoder(nn.Module):
    def __init__(self,leaky=0.1):
        super(Based_Encoder, self).__init__()
        self.leaky = leaky
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 100, kernel_size = 1, padding='valid'),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(100),  
            #nn.LeakyReLU(self.leaky),
            #nn.Dropout(0.6)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 100, out_channels = 100, kernel_size = 1,padding='valid'),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(100),  
            #nn.LeakyReLU(self.leaky),
            #nn.Dropout(0.6)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 100, out_channels = 50, kernel_size = 1,padding='valid'),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(50),  
            #nn.LeakyReLU(self.leaky),
            #nn.Dropout(0.6)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = 50, out_channels = 30, kernel_size = 1,padding='valid'),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(30),  
            #nn.LeakyReLU(self.leaky),
            #nn.Dropout(0.6)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels = 30, out_channels = 30, kernel_size = 1,padding='valid'),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(30),  
           # nn.LeakyReLU(self.leaky),
            #nn.Dropout(0.6)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        #x = self.conv5(x)
        return x

class Based_Decoder(nn.Module):
    def __init__(self,leaky=0.1):
        super(Based_Decoder, self).__init__()
        self.leaky = leaky
        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 30, out_channels = 30, kernel_size = 1, padding=0),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(30),  
            #nn.LeakyReLU(self.leaky),
            #nn.Dropout(0.6)
        )
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 30, out_channels = 50, kernel_size = 1, padding=0),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(50),  
            #nn.LeakyReLU(self.leaky),
            #nn.Dropout(0.6)
        )
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 50, out_channels = 100, kernel_size = 1, padding=0),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(100),  
            #nn.LeakyReLU(self.leaky),
            #nn.Dropout(0.6)
        )
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 100, out_channels = 100, kernel_size = 1, padding=0),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(100),  
            #nn.LeakyReLU(self.leaky),
            #nn.Dropout(0.6)
        )
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 100, out_channels = 3, kernel_size = 1, padding=0),
            nn.Upsample(scale_factor=2),
            #nn.Dropout(0.6)
        )
    
    def forward(self, x):
        #x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.tconv4(x)
        x = self.tconv5(x)
        return x


def loadBased_AutoEncoder_More(filepath):
    """[加载LeNet网络模型]

    Args:
        filepath ([str]): [LeNet的预训练模型所在的位置]

    Returns:
        [type]: [返回一个预训练的LeNet]
    """
    checkpoint = torch.load(filepath,map_location='cpu')
    model = Based_AutoEncoder_More()
    model.load_state_dict(checkpoint['state_dict'])  # 加载网络权重参数
    return model