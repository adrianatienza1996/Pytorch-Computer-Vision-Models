import torch
import torch.nn as nn
from math import ceil
device = "cuda" if torch.cuda.is_available() else "cpu"

base_model = [
    # Expansion, #Channels, #Layers, #Downsizing(True, False), Kernel_Size, Padding
    [1, 16, 1, 0, 3, 1],
    [6, 24, 2, 1, 3, 1],
    [6, 40, 2, 1, 5, 2],
    [6, 80, 3, 1, 3, 1],
    [6, 112, 3, 0, 5, 2],
    [6, 192, 4, 1, 5, 2],
    [6, 320, 1, 0, 3, 1]]

phi_values = {
    "b0": (0, 224, 0.2),
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 330, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5)}


class CNNBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, group=1):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(c_in,
                             c_out,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding,
                             groups=group,
                             bias=False)

        self.batchnorm = nn.BatchNorm2d(c_out)
        self.activation = nn.SiLU()

    def forward(self, x):
        h = self.cnn(x)
        h = self.batchnorm(h)
        return self.activation(h)


class SqueezeExcitation(nn.Module):
    def __init__(self, c_in, redution=4):
        super(SqueezeExcitation, self).__init__()
        self.hidden_channels = redution
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_in,
                      self.hidden_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.SiLU(),
            nn.Conv2d(self.hidden_channels,
                      c_in,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.Sigmoid())

    def forward(self, x):
        return self.se(x) * x


class InvertedResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, expand_ratio, survival_prob):
        super(InvertedResidualBlock, self).__init__()
        self.channels_h = c_out * expand_ratio
        self.survival_prob = survival_prob
        print(c_in != c_out)
        self.skip_sto_depth = stride == 2 or c_in != c_out

        self.conv = nn.Sequential(
            CNNBlock(c_in,
                     self.channels_h,
                     kernel_size=1,
                     stride=1,
                     padding=0),

            CNNBlock(self.channels_h,
                     self.channels_h,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     group=self.channels_h),

            SqueezeExcitation(self.channels_h),

            CNNBlock(self.channels_h,
                     c_out,
                     kernel_size=1,
                     stride=1,
                     padding=0))

    def stochastic_depth(self, x):
        if not self.training or self.skip_sto_depth:
            return x

        binary_tensor = (torch.rand(x.shape[0], 1, 1, 1) < self.survival_prob).to(device)

        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, x):
        h = self.conv(x)
        print(h.shape)
        print(self.skip_sto_depth)
        if self.skip_sto_depth:
            return h


        return self.stochastic_depth(h) + x


class MultiTask_EfficientNet(nn.Module):
    def __init__(self, version, alpha=1.2, beta=1.1):
        super(MultiTask_EfficientNet, self).__init__()
        phi, res, surv_prob = phi_values[version]
        self.surv_prob = 1 - surv_prob
        self.width_factor = beta**phi
        self.depth_factor = alpha**phi
        self.first_conv = CNNBlock(3,
                                   int(ceil(32 * self.width_factor)),
                                   kernel_size=3,
                                   stride=2,
                                   padding=1)
        self.core = self.create_net()
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.age_classifier = nn.Sequential(
            nn.Dropout(1 - self.surv_prob),
            nn.Linear(int(ceil(1280 * self.width_factor)), 9))

        self.race_classifier = nn.Sequential(
            nn.Dropout(1 - self.surv_prob),
            nn.Linear(int(ceil(1280 * self.width_factor)), 7))

        self.gender_classifier = nn.Sequential(
            nn.Dropout(1 - self.surv_prob),
            nn.Linear(int(ceil(1280 * self.width_factor)), 1))

    def create_net (self):
        last_num_channels = int(ceil(32 * self.width_factor))
        net = []
        for expansion,  c, num_layers, downsizing, kernel_size, padding in base_model:
                num_channels = int(ceil(c * self.width_factor))
                num_layers = int(ceil(num_layers * self.depth_factor))
                for i in range(num_layers - 1):
                    net.append(InvertedResidualBlock(last_num_channels,
                                                     num_channels,
                                                     kernel_size,
                                                     stride=1,
                                                     padding=padding,
                                                     expand_ratio=expansion,
                                                     survival_prob=self.surv_prob))

                    last_num_channels = num_channels

                net.append(InvertedResidualBlock(last_num_channels,
                                                 num_channels,
                                                 kernel_size,
                                                 stride=1 + downsizing,
                                                 padding=padding,
                                                 expand_ratio=expansion,
                                                 survival_prob=self.surv_prob
                                                 ))

                last_num_channels = num_channels


        net.append(CNNBlock(
            last_num_channels,
            int(ceil(1280 * self.width_factor)),
            kernel_size=1,
            stride=1,
            padding=0))

        print(len(net))
        return nn.Sequential(*net)

    def forward(self, x):
        h = self.first_conv(x)
        h = self.core(h)
        h = self.pool(h).squeeze()
        age = self.age_classifier(h)
        race = self.race_classifier(h)
        gender = self.gender_classifier(h)
        return age, race, gender








