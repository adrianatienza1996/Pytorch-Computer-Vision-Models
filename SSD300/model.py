import numpy as np
import torch
import torchvision
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {'conv4':  (512, 4, [1.0, 2.0, 0.5]),
          'conv7':  (1024, 6, [1.0, 2.0, 3.0, 0.5, 0.333]),
          'conv8':  (512, 6, [1.0, 2.0, 3.0, 0.5, 0.333]),
          'conv9':  (256, 6, [1.0, 2.0, 3.0, 0.5, 0.333]),
          'conv10': (256, 4, [1.0, 2.0, 0.5]),
          'conv11': (256, 4, [1.0, 2.0, 0.5])}

SCALES = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]

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
        conv4_features, h = self.block4(h)
        h = self.block5(h)
        return conv4_features, self.final_block(h)

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
        print("Pretrained weights are loaded")\



class Aux_Conv_Block1(nn.Module):
    def __init__(self, c_in, c_out, inner_dim):
        super(Aux_Conv_Block1, self).__init__()
        self.net = nn.Sequential(
            Conv2D_RELU(c_in=c_in, c_out=inner_dim, kernel_size=1, stride=1, padding=0),
            Conv2D_RELU(c_in=inner_dim, c_out=c_out, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        return self.net(x)


class Aux_Conv_Block2(nn.Module):
    def __init__(self, c_in, c_out, inner_dim):
        super(Aux_Conv_Block2, self).__init__()
        self.net = nn.Sequential(
            Conv2D_RELU(c_in=c_in, c_out=inner_dim, kernel_size=1, stride=1, padding=0),
            Conv2D_RELU(c_in=inner_dim, c_out=c_out, kernel_size=3, stride=1, padding=0)
        )

    def forward(self, x):
        return self.net(x)


class Aux_Conv(nn.Module):
    def __init__(self):
        super(Aux_Conv, self).__init__()
        self.block1 = Aux_Conv_Block1(1024, 512, 256)
        self.block2 = Aux_Conv_Block1(512, 256, 128)
        self.block3 = Aux_Conv_Block2(256, 256, 128)
        self.block4 = Aux_Conv_Block2(256, 256, 128)

    def forward(self, x):
        features_8 = self.block1(x)
        features_9 = self.block2(features_8)
        features_10 = self.block3(features_9)
        features_11 = self.block4(features_10)

        return features_8, features_9, features_10, features_11


class Predictor_Block(nn.Module):
    def __init__(self, c_in, num_priors, num_classes):
        super(Predictor_Block, self).__init__()
        self.num_classes = num_classes
        self.loc_predictor = nn.Conv2d(c_in, int(num_priors * 4),
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        self.label_predictor = nn.Conv2d(c_in, int(num_priors * num_classes),
                                         kernel_size=3,
                                         stride=1,
                                         padding=1)

    def forward(self, x):
        locs = self.loc_predictor(x)
        locs = locs.permute(0, 2, 3, 1).contiguous()
        locs = locs.view(x.shape[0], -1, 4)

        labels = self.label_predictor(x)
        labels = labels.permute(0, 2, 3, 1).contiguous()
        labels = labels.view(x.shape[0], -1, self.num_classes)

        return locs, labels


class Predictor(nn.Module):
    def __init__(self, num_classes):
        super(Predictor, self).__init__()
        self.predictor4 = Predictor_Block(CONFIG['conv4'][0], CONFIG['conv4'][1], num_classes)
        self.predictor7 = Predictor_Block(CONFIG['conv7'][0], CONFIG['conv7'][1], num_classes)
        self.predictor8 = Predictor_Block(CONFIG['conv8'][0], CONFIG['conv8'][1], num_classes)
        self.predictor9 = Predictor_Block(CONFIG['conv9'][0], CONFIG['conv9'][1], num_classes)
        self.predictor10 = Predictor_Block(CONFIG['conv10'][0], CONFIG['conv10'][1], num_classes)
        self.predictor11 = Predictor_Block(CONFIG['conv11'][0], CONFIG['conv11'][1], num_classes)

    def forward(self, features4, features7, features8, features9, features10, features11):
        locs4, labels4 = self.predictor4(features4)
        locs7, labels7 = self.predictor7(features7)
        locs8, labels8 = self.predictor8(features8)
        locs9, labels9 = self.predictor9(features9)
        locs10, labels10 = self.predictor10(features10)
        locs11, labels11 = self.predictor11(features11)

        locs = torch.cat([locs4, locs7, locs8, locs9, locs10, locs11], dim=1)
        labels = torch.cat([labels4, labels7, labels8, labels9, labels10, labels11], dim=1)

        return locs, labels


class SSD_300(nn.Module):
    def __init__(self, num_classes):
        super(SSD_300, self).__init__()
        self.vgg = VGGBase()
        self.aux_conv = Aux_Conv()
        self.predictor = Predictor(num_classes)

    def forward(self, x):
        features4, features7 = self.vgg(x)
        features8, features9, features10, features11 = self.aux_conv(features7)
        locs, labels = self.predictor(features4, features7, features8, features9, features10, features11)
        return locs, labels

    def get_prior_boxes(self):
        prior_boxes = []

        def get_coordinates(i, j, scale, aspect_ratio, dim):
            return [(i + 0.5) / dim, (j + 0.5) / dim, scale * np.sqrt(aspect_ratio), scale / np.sqrt(aspect_ratio)]

        def geometric_mean(x, y):
            return np.sqrt(x * y)

        for idx, (_, config) in enumerate(CONFIG.items()):

            scale = SCALES[idx]
            dim = config[2]
            x, y = np.meshgrid(np.arange(dim), np.arange(dim))

            for ratio in config[3]:
                prior_boxes.extend([get_coordinates(i, j, scale, ratio, dim) for i, j in zip(np.ravel(x), np.ravel(y))])

            try:
                add_scale = geometric_mean(SCALES[idx], SCALES[idx + 1])

            except IndexError:
                add_scale = 1.

            prior_boxes.extend([get_coordinates(i, j, add_scale, 1, dim) for i, j in zip(np.ravel(x), np.ravel(y))])

        prior_boxes = torch.tensor(prior_boxes).float().to(device)
        prior_boxes.clamp_(0, 1)
        return prior_boxes

    







