import torch 
import torch.nn as nn

class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, final=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = PixelNorm()
        self.act = nn.LeakyReLU(0.2) if not final else nn.Tanh()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Block(3, 64)
        self.conv2 = Block(64, 128)

        self.upscale = nn.Upsample(scale_factor=4, mode='nearest')
        self.final_conv= Block(128, 3, final=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upscale(x)
        x = self.final_conv(x)
        return x
    