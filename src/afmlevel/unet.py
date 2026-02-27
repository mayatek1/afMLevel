import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(
        self, n_channels, filtersize1=9, filtersize=9, leakyrelu=False, dropoutprob=0
    ):
        super().__init__()

        padding1 = (filtersize1 - 1) // 2
        padding = (filtersize - 1) // 2

        activation = nn.LeakyReLU(inplace=True) if leakyrelu else nn.ReLU(inplace=True)

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, filtersize, padding=padding),
                nn.BatchNorm2d(out_channels),
                activation,
                nn.Dropout(dropoutprob),
                nn.Conv2d(out_channels, out_channels, filtersize, padding=padding),
                nn.BatchNorm2d(out_channels),
                activation,
                nn.Dropout(dropoutprob),
            )

        def double_conv1(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, filtersize1, padding=padding1),
                nn.BatchNorm2d(out_channels),
                activation,
                nn.Dropout(dropoutprob),
                nn.Conv2d(out_channels, out_channels, filtersize1, padding=padding1),
                nn.BatchNorm2d(out_channels),
                activation,
                nn.Dropout(dropoutprob),
            )

        self.dc1 = double_conv1(n_channels, 16)
        self.dc2 = double_conv(16, 32)
        self.dc3 = double_conv(32, 64)
        self.dc4 = double_conv(64, 128)
        self.dc5 = double_conv(128, 256)
        self.dc6 = double_conv(256, 512)
        self.dc7 = double_conv(512, 1024)

        # Upsampling
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dc8 = double_conv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dc9 = double_conv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dc10 = double_conv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dc11 = double_conv(128, 64)
        self.up5 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dc12 = double_conv(64, 32)
        self.up6 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dc13 = double_conv(32, 16)

        self.final = nn.Conv2d(16, 1, 1)
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.dc1(x)
        x2 = self.dc2(self.max_pool(x1))
        x3 = self.dc3(self.max_pool(x2))
        x4 = self.dc4(self.max_pool(x3))
        x5 = self.dc5(self.max_pool(x4))
        x6 = self.dc6(self.max_pool(x5))
        x7 = self.dc7(self.max_pool(x6))

        x = self.up1(x7)
        x = self.dc8(torch.cat([x6, x], dim=1))
        x = self.up2(x)
        x = self.dc9(torch.cat([x5, x], dim=1))
        x = self.up3(x)
        x = self.dc10(torch.cat([x4, x], dim=1))
        x = self.up4(x)
        x = self.dc11(torch.cat([x3, x], dim=1))
        x = self.up5(x)
        x = self.dc12(torch.cat([x2, x], dim=1))
        x = self.up6(x)
        x = self.dc13(torch.cat([x1, x], dim=1))

        return self.final(x)
