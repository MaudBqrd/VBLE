import torch.nn as nn
import torch.nn.functional as F
import torch


class LightEncoder(nn.Module):
    """
    Encoder (inference model) of a VAE
    """

    def __init__(self, in_channels: int, N: int = 32, M: int = 64):
        """
        Parameters
        ----------
        in_channels: number of input channels
        N: number of channels of the latent variable
        M: number of hidden channels in the encoder and decoder
        """
        super(LightEncoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, N, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(N),
            nn.LeakyReLU(),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(N),
            nn.LeakyReLU(),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(N),
            nn.LeakyReLU(),
            nn.Conv2d(N, 2*M, kernel_size=5, stride=2, padding=2)
        )

    def forward(self, x):
        x = self.layers(x)
        mu_z, logsd_z = x.chunk(2, dim=1)
        return {
            "mu_z": mu_z,
            "logsd_z": logsd_z
        }


class LightEncoderFCBottleneck(nn.Module):
    """
    Encoder (inference model) of a VAE
    """

    def __init__(self, in_channels: int, N: int = 32, M: int = 64):
        """
        Parameters
        ----------
        in_channels: number of input channels
        N: number of channels of the latent variable
        M: number of hidden channels in the encoder and decoder
        """
        super(LightEncoderFCBottleneck, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, N, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
        )
        self.bottleneck = nn.Linear(N*4*4, 2*M)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        mu_z, logsd_z = x.chunk(2, dim=1)
        return {
            "mu_z": mu_z,
            "logsd_z": logsd_z
        }

## ResNet Encoder

class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Enc(nn.Module):

    def __init__(self, in_channels, N, M, num_Blocks=[2,2,2,2]):
        super().__init__()
        self.in_planes = 64
        self.z_dim = M
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
        self.last_conv = nn.Conv2d(512, 2*M, kernel_size=3, stride=1, padding=1, bias=False)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.last_conv(x)
        mu_z, logvar_z = x.chunk(2, dim=1)
        return {
            "mu_z": mu_z,
            "logsd_z": logvar_z
        }
