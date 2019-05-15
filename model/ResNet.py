import torch.nn as nn
import torch.nn.functional as F


#def conv3x3(in_planes, out_planes):
#    """3x3 convolution with padding"""
#    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
#                     padding=1, bias=False)
###

class BasicBlock(nn.Module):
    # expansion = 1#

    def __init__(self, in_planes, out_planes, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn = nn.BatchNorm2d(out_planes)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = downsample


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Encoder(nn.Module):
    def __init__(self, nIn, nHidden):
        super(Encoder, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, num_layers=2)

    def forward(self, x):#x:w(t) * b * h(c)
        self.rnn.flatten_parameters()
        _, (h_n, c_n) = self.rnn(x)
        out = h_n[-1]

        return out


class ResNet(nn.Module):
    def __init__(self, nchannel, nhidden ):
        super(ResNet, self).__init__()
        self.inplanes = 128

        self.conv1 = self.convRelu(nchannel, 64)
        self.conv2 = self.convRelu(64, 128)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.res1 = self._make_layer(256, 1)
        self.conv3 = self.convRelu(256, 256)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.res2 = self._make_layer(256, 2)
        self.conv4 = self.convRelu(256, 256)
        # self.maxpool3 = nn.MaxPool2d((1, 2), (1, 2))
        self.maxpool3 = nn.MaxPool2d((2, 1), (2, 1))
        self.res3 = self._make_layer(512, 5)
        self.conv5 = self.convRelu(512, 512)
        self.res4 = self._make_layer(512, 3)
        self.conv6 = self.convRelu(512, 512) #b * c(input_size) * h * w
        self.rnn = Encoder(512, nhidden)
        self.lstm = nn.LSTM(nhidden, nhidden, num_layers=2)
        # res = nn.Sequential()
        # self.inplanes = 128
        # self.conv1 = conv3x3(3, 64, stride=1)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(64, 128, stride=1)
        # self.bn2 = nn.BatchNorm2d(128)
        #
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.layer1 = self._make_layer(block, 256, 1)
        # self.conv3 = conv3x3(256, 256)
        # self.bn3 = nn.BatchNorm2d(256)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.layer2 = self._make_layer(block, 256, 2)
        # self.conv4 = conv3x3(256, 256)
        # self.bn4 = nn.BatchNorm2d(256)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        # self.layer3 = self._make_layer(block, 512, 5)
        # self.conv5 = conv3x3(512, 512)
        # self.bn5 = nn.BatchNorm2d(512)
        # self.layer4 = self._make_layer(block, 512, 3)
        # self.conv6 = conv3x3(512, 512)
        # self.bn6 = nn.BatchNorm2d(512)

    def forward(self, x):
        #print(x)
        feature = self.conv1(x)
        #print(featrue.type)
        feature = self.conv2(feature)
        feature = self.maxpool1(feature)
        feature = self.res1(feature)
        feature = self.conv3(feature)
        feature = self.maxpool2(feature)
        feature = self.res2(feature)
        feature = self.conv4(feature)
        feature = self.maxpool3(feature)
        feature = self.res3(feature)
        feature = self.conv5(feature)
        feature = self.res4(feature)
        feature = self.conv6(feature)
        feature_map = feature #b * c(input_size) * h * w

        imgH = feature.size(2)
        feature = F.max_pool2d(feature, (imgH, 1), (imgH, 1))
        #print(feature.shape)
        feature = feature.squeeze(2) # b * c * w
        feature = feature.permute(2, 0, 1)


        out = self.rnn(feature).unsqueeze(0)
        self.lstm.flatten_parameters()
        _, hidden_state = self.lstm(out)

        return hidden_state, feature_map

        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        #
        # x = self.maxpool1(x)
        # x = self.layer1(x)
        #
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.maxpool2(x)
        # x = self.layer2(x)
        #
        # x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.maxpool3(x)
        # x = self.layer3(x)
        #
        # x = self.conv5(x)
        # x = self.bn5(x)
        # x = self.layer4(x)
        #
        # x = self.con6(x)
        # x = self.bn6(x)


    
    #def conv3x3(self, in_planes, out_planes):
    #    """3x3 convolution with padding"""
    #    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)




    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes ,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def convRelu(self, in_planes, out_planes):
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
                             nn.BatchNorm2d(out_planes),
                             nn.ReLU(inplace=True))
