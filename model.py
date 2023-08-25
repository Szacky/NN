import warnings
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings("ignore")


class Model(nn.Module):
    def __init__(self, input_channel, ndim, nclass):
        super(Model, self).__init__()
        self.tag = "base"
        self.ndim = ndim
        self.nclass = nclass

        self.input_conv = nn.Conv2d(
            input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(64)

        input_channel = 64
        hidden_channels = [128, 512, 1024]
        self.hidden_convs = []
        for i in range(3):
            output_channel = hidden_channels[i]
            self.hidden_convs.append(nn.Conv2d(input_channel, output_channel,
                                               kernel_size=3, stride=2, bias=False, padding=1))
            self.hidden_convs.append(nn.BatchNorm2d(output_channel))
            self.hidden_convs.append(nn.ReLU())
            input_channel = output_channel
        self.hidden_convs = nn.Sequential(*self.hidden_convs)

        self.fc = nn.Linear(input_channel, nclass)
        return

    def forward(self, x):
        out = F.relu(self.input_bn(self.input_conv(x)))
        out = self.hidden_convs(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def rep_forward(self, x):
        out = F.relu(self.input_bn(self.input_conv(x)))
        out = self.hidden_convs(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, input_channel, ndim, num_classes, block=BasicBlock, num_blocks=[2, 2, 2, 2]):
        super(ResNet, self).__init__()
        self.tag = "resnet"
        self.in_planes = 64
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def rep_forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out
