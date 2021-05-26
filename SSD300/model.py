from torch import nn
import torchvision


class Conv2D_RELU(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation=1):
        super(Conv2D_RELU, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=c_in,
                      out_channels=c_out,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation),
            nn.ReLU())

    def forward(self, x):
        return self.net(x)


class VGG_Block(nn.Module):
    def __init__(self, c_in, c_out,
                 kernel_size, stride, padding, num_conv_layers,
                 ceil_mode=False, final_block=False, return_features=False):

        super(VGG_Block, self).__init__()
        net = [Conv2D_RELU(c_in, c_out, kernel_size, stride, padding)]
        for _ in range(num_conv_layers - 1):
            net.append(Conv2D_RELU(c_out, c_out, kernel_size, stride, padding))

        self.net = nn.Sequential(*net)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=ceil_mode) if final_block \
            else nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode)

        self.return_features = return_features

    def forward(self, x):
        features = self.net(x)
        if self.return_features:
            return features, self.maxpool(features)

        return self.maxpool(features)


class VGGBase(nn.Module):
    def __init__(self):
        super(VGGBase, self).__init__()
        self.block1 = VGG_Block(3, 64,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                num_conv_layers=2)

        self.block2 = VGG_Block(64, 128,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                num_conv_layers=2)

        self.block3 = VGG_Block(128, 256,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                num_conv_layers=3,
                                ceil_mode=True)

        self.block4 = VGG_Block(256, 512,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                num_conv_layers=3,
                                return_features=True)

        self.block5 = VGG_Block(512, 512,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                num_conv_layers=3, final_block=True)

        self.final_block = nn.Sequential(
            Conv2D_RELU(512, 1024, kernel_size=3, stride=1, padding=6, dilation=6),
            Conv2D_RELU(1024, 1024, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        conv4_3_features, h = self.block4(h)
        h = self.block5(h)
        return conv4_3_features, self.final_block(h)

    def load_pretrained_weights(self):
        # Params of the model
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Params of pretrained model
        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        for param_name1, param_name2 in zip(param_names[:-4], pretrained_param_names[:-6]):
            state_dict[param_name1] = pretrained_state_dict[param_name2]

        # Number of weights/bias is lower in our custom VGG model. Just picking some of the pretrained classifier

        x, y, _, _ = state_dict[param_names[-4]].shape
        state_dict[param_names[-4]] = pretrained_state_dict[pretrained_param_names[-6]][:x, :y]
        state_dict[param_names[-3]] = pretrained_state_dict[pretrained_param_names[-5]][:x]

        x, y, _, _ = state_dict[param_names[-2]].shape
        state_dict[param_names[-2]] = pretrained_state_dict[pretrained_param_names[-4]][:x, :y]
        state_dict[param_names[-1]] = pretrained_state_dict[pretrained_param_names[-3]][:x]

        print("Pretrained weights are loaded")

