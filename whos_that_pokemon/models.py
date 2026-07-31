import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.out_conv = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))  
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e3], dim=1))   
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out_conv(d1)


class PokemonClassifier(nn.Module):
    def __init__(self, num_classes, in_ch=1, base=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(base, base * 2, 3, padding=1), nn.BatchNorm2d(base * 2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(base * 2, base * 4, 3, padding=1), nn.BatchNorm2d(base * 4), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(base * 4, base * 8, 3, padding=1), nn.BatchNorm2d(base * 8), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base * 8, num_classes)

def forward(self, x):
    x = self.features(x)
    x = self.global_pool(x).flatten(1)
    
    return self.fc(x)


def check_unet_shape(out_size=128):
    model = UNet()
    out = model(torch.randn(2, 3, out_size, out_size))
    assert out.shape == (2, 1, out_size, out_size)
    print("U-Net output shape OK:", out.shape)


def check_classifier_shape(num_classes, out_size=128):
    clf = PokemonClassifier(num_classes=num_classes)
    out = clf(torch.randn(2, 1, out_size, out_size))
    assert out.shape == (2, num_classes)
    print("Classifier output shape OK:", out.shape)


if __name__ == "__main__":
    check_unet_shape()
    check_classifier_shape(num_classes=151)