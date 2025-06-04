# uhrnet_architecture.py
import torch
import torch.nn as nn
from torch import cat

class Branch0(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        return self.bn(self.conv(x))

class Branch1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        return self.bn(self.conv(x))

class Branch2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(x))

class Branch3(Branch2):
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn2(self.conv2(x))
        return x

class Branch4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch, 3, padding=2, dilation=2)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.conv4 = nn.Conv2d(out_ch, out_ch, 3, padding=5, dilation=5)
        self.bn4 = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return self.bn4(self.conv4(x))

class ResB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.b0 = Branch0(in_ch, out_ch)
        self.b1 = Branch1(in_ch, out_ch // 4)
        self.b2 = Branch2(in_ch, out_ch // 4)
        self.b3 = Branch3(in_ch, out_ch // 4)
        self.b4 = Branch4(in_ch, out_ch // 4)
        self.relu = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.b0(x) + cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1))

class DownB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.res = ResB(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        x1 = self.res(x)
        return self.pool(x1), x1

class UpB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 3, 2, 1, output_padding=1)
        self.res = ResB(out_ch*2, out_ch)
    def forward(self, x, skip):
        return self.res(cat([self.up(x), skip], dim=1))

class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x):
        return self.conv(x)

class UHRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DownB(1, 64)
        self.down2 = DownB(64, 128)
        self.down3 = DownB(128, 256)
        self.down4 = DownB(256, 512)
        self.bottleneck = ResB(512, 1024)
        self.up1 = UpB(1024, 512)
        self.up2 = UpB(512, 256)
        self.up3 = UpB(256, 128)
        self.up4 = UpB(128, 64)
        self.out = Outconv(64, 1)

    def forward(self, x):
        x1, s1 = self.down1(x)
        x2, s2 = self.down2(x1)
        x3, s3 = self.down3(x2)
        x4, s4 = self.down4(x3)
        x5 = self.bottleneck(x4)
        x = self.up1(x5, s4)
        x = self.up2(x, s3)
        x = self.up3(x, s2)
        x = self.up4(x, s1)
        return self.out(x)
