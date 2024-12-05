import torch
import torch.nn as nn
import torch.nn.functional as F

class LightDecoder(nn.Module):
    """
    Decoder (generative model) of a VAE
    """

    def __init__(self, in_channels: int, N:int = 32, M:int = 64, gamma: float = 1.):
        """
        Parameters
        ----------
        in_channels: number of input channels
        N: number of channels of the latent variable
        M: number of hidden channels in the encoder and decoder
        gamma: standard deviation of the decoder (p(x|z) = N(mu_\theta(z), \gamma^2 I)), to fix the trade off between
               the KL and the data fidelity term. Put gamma='variable' to learn it as a nn parameters
        """
        super(LightDecoder, self).__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(M, N, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(N),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(N),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(N),
            nn.LeakyReLU(),
            # nn.Conv2d(N, in_channels, kernel_size=5, stride=1, padding=2),
            nn.ConvTranspose2d(N, in_channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        )

        self.learn_gamma = gamma == "variable"
        if self.learn_gamma:
            self.gamma_x = nn.Parameter(torch.ones(1))
        else:
            self.gamma_x = nn.Parameter(float(gamma) * torch.ones(1), requires_grad=False)

    def forward(self, z):
        x = self.layers(z)
        return {
            "x_rec": x,
            "gamma_x": self.gamma_x
        }


class LightDecoderFCBottleneck(nn.Module):
    """
    Decoder (generative model) of a VAE
    """

    def __init__(self, in_channels: int, N:int = 32, M:int = 64, gamma: float = 1.):
        """
        Parameters
        ----------
        in_channels: number of input channels
        N: number of channels of the latent variable
        M: number of hidden channels in the encoder and decoder
        gamma: standard deviation of the decoder (p(x|z) = N(mu_\theta(z), \gamma^2 I)), to fix the trade off between
               the KL and the data fidelity term. Put gamma='variable' to learn it as a nn parameters
        """
        super(LightDecoderFCBottleneck, self).__init__()

        self.first_layer = nn.Sequential(
            nn.Linear(M, N*4*4),
            nn.LeakyReLU(),
        )

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(N, in_channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        )

        self.learn_gamma = gamma == "variable"
        if self.learn_gamma:
            self.gamma_x = nn.Parameter(torch.ones(1))
        else:
            self.gamma_x = nn.Parameter(float(gamma) * torch.ones(1), requires_grad=False)

    def forward(self, z):
        z = self.first_layer(z)
        z = z.view(z.size(0), -1, 4, 4)
        x = self.layers(z)
        return {
            "x_rec": x,
            "gamma_x": self.gamma_x
        }



## ResNet Decoder

class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out
    

class ResNet18Dec(nn.Module):

    def __init__(self, in_channels: int, N:int = 32, M:int = 64, gamma: float = 1., num_Blocks=[2,2,2,2]):
        super().__init__()
        self.in_planes = 512
        self.z_dim = M
        
        self.learn_gamma = gamma == "variable"
        if self.learn_gamma:
            self.gamma_x = nn.Parameter(torch.ones(1))
        else:
            self.gamma_x = nn.Parameter(float(gamma) * torch.ones(1), requires_grad=False)

        self.first_conv = nn.ConvTranspose2d(self.z_dim, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, in_channels, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.first_conv(z)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.conv1(x)
        return {
            "x_rec": x,
            "gamma_x": self.gamma_x
        }