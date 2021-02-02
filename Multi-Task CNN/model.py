import torch
import torch.nn as nn
device = "cuda" if torch.cuda.is_available() else "cpu"

class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=torch.nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class ResBlock (nn.Module):

    def __init__ (self, ni, no, kernel_size=3, stride=1):
        super(ResBlock, self).__init__()
        self.kernel_size = 3
        self.stride = 1
        self.no = no
        self.conv1 = Conv2dSame(ni, no, kernel_size=kernel_size)
        self.activation1 = nn.ReLU()
        self.batch1 = nn.BatchNorm2d(no)

        self.conv2 = Conv2dSame(no, no, kernel_size=kernel_size)
        self.activation2 = nn.ReLU()

        self.identity = Conv2dSame(ni, no, kernel_size=1)

    def forward(self, x):
        identity = self.identity(x)

        h = self.conv1(x)
        h = self.activation1(h)
        h = self.batch1(h)
        h = self.conv2(h)

        h = h + identity
        return self.activation2(h)


class ResNet18_Classifier (nn.Module):

    def __init__(self):
        super(ResNet18_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding = (3, 3))
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding= (1, 1))
        self.res_block1 = ResBlock(64, 64)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res_block2 = ResBlock(64, 128)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res_block3 = ResBlock(128, 256)
        self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res_block4 = ResBlock(256, 512)
        self.avg_pool1 = nn.AvgPool2d (kernel_size=7)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(512, 1000)

        self.age_classifier = nn.Sequential(
                    nn.Linear(1000, 9),
                    nn.Softmax(dim = 1))

        self.race_classifier = nn.Sequential(
            nn.Linear(1000, 7),
            nn.Softmax(dim = 1))

        self.gender_classifier = nn.Sequential(
            nn.Linear(1000, 1),
            nn.Sigmoid())

    def forward (self, x):
        h = self.conv1(x)
        h = self.max_pool1(h)
        h = self.res_block1(h)
        h = self.max_pool2(h)
        h = self.res_block2(h)
        h = self.max_pool3(h)
        h = self.res_block3(h)
        h = self.max_pool4(h)
        h = self.res_block4(h)
        h = self.avg_pool1(h)
        h = self.flatten(h)

        output = self.linear1(h)
        age = self.age_classifier(output)
        race = self.race_classifier(output)
        gender = self.gender_classifier(output)

        return age, race, gender