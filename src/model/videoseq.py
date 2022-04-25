import torch
import torch.nn as nn
import torch.nn.functional as F


from model.SPP import ASPP_simple, ASPP
from model.convlstm import ConvLSTM


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class VideoSOD(nn.Module):
    def __init__(self, nInputChannels, n_classes, os, img_backbone_type, num_layers, batch_first, bias, device):
        super(VideoSOD, self).__init__()

        self.inplanes = 64
        self.os = os
        self.hidden_dim = 32
        self.kernel_size = 3
        self.padding = [1, 2]
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.dilation = [1, 2]
        self.device = device

        if os == 16:
            aspp_rates = [1, 6, 12, 18]
        elif os == 8 or os == 32:
            aspp_rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        if os == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
        elif os == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
        elif os == 32:
            strides = [1, 2, 2, 2]
            rates = [1, 1, 1, 1]
        else:
            raise NotImplementedError

        assert img_backbone_type == 'resnet101'

        # Modules
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layers = [3, 4, 23, 3]

        self.layer1 = self._make_layer(64, layers[0], stride=strides[0], rate=rates[0])
        self.layer2 = self._make_layer(128, layers[1], stride=strides[1], rate=rates[1])
        self.layer3 = self._make_layer(256, layers[2], stride=strides[2], rate=rates[2])
        self.layer4 = self._make_layer(512, layers[3], stride=strides[3], rate=rates[3])

        asppInputChannels = 2048
        asppOutputChannels = 256
        lowInputChannels =  256
        lowOutputChannels = 48

        # low_level_features to 48 channels
        #self.conv2 = nn.Conv2d(lowInputChannels, lowOutputChannels, 1, bias=False)
        #self.bn2 = nn.BatchNorm2d(lowOutputChannels)

        self.aspp = ASPP(asppInputChannels, asppOutputChannels, aspp_rates)

        self.convLSTM1 = ConvLSTM(asppOutputChannels, self.hidden_dim, self.kernel_size, self.padding[0], self.num_layers,
                                 self.batch_first, self.dilation[0], self.bias, self.device)

        self.convLSTM2 = ConvLSTM(asppOutputChannels, self.hidden_dim, self.kernel_size, self.padding[1], self.num_layers,
                                 self.batch_first, self.dilation[1], self.bias, self.device)

        self.last_convs = nn.Sequential(
            nn.Conv2d(2*self.hidden_dim, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
        )

    def _make_layer(self, planes, blocks, stride=1, rate=1):

        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, sequences):
        img_features_list = []
        low_level_features_list = []
        seq_len = sequences.shape[1]
        for t in range(seq_len):
            img = sequences[:, t, :, :, :]

            x = self.conv1(img)
            x = self.bn1(x)
            x = self.relu(x)
            conv1_feat = x

            x = self.maxpool(x)
            x = self.layer1(x)
            layer1_feat = x
            low_level_features = x
            #low_level_features = self.conv2(low_level_features)
            #low_level_features = self.bn2(low_level_features)
            #low_level_features_list.append(low_level_features)


            x = self.layer2(x)
            layer2_feat = x

            x = self.layer3(x)
            layer3_feat = x

            x = self.layer4(x)
            layer4_feat = x

            if self.os == 32:
                x = F.upsample(x, scale_factor=4, mode='bilinear', align_corners=True)

            aspp = self.aspp(x)
            # img_feat_lst.append(x)

            img_features_list.append(aspp)
        x = torch.stack(img_features_list, dim=1)
        convLSTM1_output = self.convLSTM1(x)
        convLSTM2_output = self.convLSTM2(x)
        layer_1 = len(convLSTM1_output) - 1
        layer_2 = len(convLSTM2_output) - 1
        x_1 = convLSTM1_output[layer_1]
        x_2 = convLSTM2_output[layer_2]


        convLSTM1_output_list1 = []
        convLSTM2_output_list1 = []
        convLSTM1_concat_list = []

        #run through the first convLSTM
        for t in range(seq_len):
            #low_level_features = low_level_features_list[t]
            convlstm_features = x_1[:, t, :, :, :]
            #convlstm_features = F.upsample(convlstm_features, low_level_features.size()[2:], mode='bilinear', align_corners=True)
            #stacked_features = torch.cat((convlstm_features, low_level_features), dim=1)
            convLSTM1_output_list1.append(convlstm_features)

        # run through the second convLSTM
        for t in range(seq_len):
            convlstm1_features = convLSTM1_output_list1[t]
            convlstm_features = x_2[:, t, :, :, :]
            #convlstm_features = F.upsample(convlstm_features, low_level_features.size()[2:], mode='bilinear', align_corners=True)
            concat_features = torch.cat((convlstm_features, convlstm1_features), dim=1)
            print("Shape of concat of both convLSTMs: {}".format(concat_features.shape))
            convLSTM2_output_list1.append(concat_features)
        x = torch.stack(convLSTM2_output_list1, dim=1)
        print("Stacked both ConvLSTM outputs: {}".format(x.shape))

        saliency_maps = []

        for t in range(seq_len):
            saliency = self.last_convs(x[:, t, :, :, :])
            saliency_maps.append(saliency)
        x = torch.stack(saliency_maps, dim=1)
        return x