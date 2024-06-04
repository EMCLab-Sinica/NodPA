import torch
import torch.nn as nn

class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class har_cnn(nn.Module):
    def __init__(self, n_channels=9, n_classes=6):
        super().__init__()
        # (batch, 9, 128) -> (batch, 18, 64)
        self.conv1 = ConvReLU(in_channels=n_channels, out_channels=18, kernel_size=(1,2), padding=(0,1))
        self.pool1 = nn.MaxPool2d(kernel_size=(1,2), padding=0)
        # (batch, 18, 64) -> (batch, 36, 32)
        self.conv2 = ConvReLU(in_channels=18, out_channels=36, kernel_size=(1,2), padding=(0,1))
        self.pool2 = nn.MaxPool2d(kernel_size=(1,2), padding=0)

        # (batch, 36, 32) -> (batch, 72, 16)
        self.conv3 = ConvReLU(in_channels=36, out_channels=72, kernel_size=(1,2), padding=(0,1))
        self.pool3 = nn.MaxPool2d(kernel_size=(1,2), padding=0)

        self.ip1 = nn.Linear(16*72, n_classes)

    def forward(self, x):
        x = torch.unsqueeze(x, 2)

        # (batch, 9, 1, 128)
        x = self.conv1(x)
        # (batch, 18, 128)
        x = self.pool1(x)
        # (batch, 18, 1, 64)
        x = self.conv2(x)
        # (batch, 36, 64)
        x = self.pool2(x)
        # (batch, 36, 1, 32)
        x = self.conv3(x)
        # (batch, 72, 16)
        x = self.pool3(x)

        x = x.view(x.size(0), 16*72)

        x = self.ip1(x,)
        return x
