import torch
import torch.nn as nn
from torch.nn import functional as F


class ConvBNRelu(nn.Module):
    """
    Conv2d + BatchNorm2d + Relu
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1, bias=False):
        super(ConvBNRelu, self).__init__()
        if padding is None:
            padding = 0
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      groups=groups,
                      bias=bias,
                      ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.seq(x)


class Stem(nn.Module):
    def __init__(self, in_channels=3, out_channels=256):
        super(Stem, self).__init__()
        self.stem = nn.Sequential(
            # input N x 3 x 299 x 299
            ConvBNRelu(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2),
            # input N x 32 x 149 x 149
            ConvBNRelu(in_channels=32, out_channels=32, kernel_size=3),
            # input N x 32 x 147 x 147
            ConvBNRelu(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            # input N x 64 x 147 x 147
            nn.MaxPool2d(kernel_size=3, stride=2),
            # input N x 64 x 73 x 73
            ConvBNRelu(in_channels=64, out_channels=80, kernel_size=1),
            # input N x 80 x 73 x 73
            ConvBNRelu(in_channels=80, out_channels=192, kernel_size=3),
            # input N x 192 x 71 x 71
            ConvBNRelu(in_channels=192, out_channels=out_channels, kernel_size=3, stride=2)
            # output N x 256 x 35 x 35
        )

    def forward(self, x):
        return self.stem(x)


class InceptionResNetA(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, scale=1.0):
        super(InceptionResNetA, self).__init__()
        self.scale = scale
        # input N x 256 x 35 x 35
        # output N x 32 x 35 x 35
        self.branch_1 = ConvBNRelu(in_channels=in_channels, out_channels=32, kernel_size=1)
        # input N x 256 x 35 x 35
        # output N x 32 x 35 x 35
        self.branch_2 = nn.Sequential(
            ConvBNRelu(in_channels=in_channels, out_channels=32, kernel_size=1),
            ConvBNRelu(in_channels=32, out_channels=32, kernel_size=3, padding=1),
        )
        # input N x 256 x 35 x 35
        # output N x 32 x 35 x 35
        self.branch_3 = nn.Sequential(
            ConvBNRelu(in_channels=in_channels, out_channels=32, kernel_size=1),
            ConvBNRelu(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            ConvBNRelu(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        )
        # input N x 96 x 35 x 35
        # output N x 256 x 35 x 35
        self.conv2d_branch = nn.Conv2d(in_channels=96, out_channels=out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        branch_all = torch.cat((x1, x2, x3), 1)
        out = self.conv2d_branch(branch_all)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class ReductionA(nn.Module):
    def __init__(self, in_channels=256, k=192, l=192, m=256, n=384):
        super(ReductionA, self).__init__()
        # input N x 256 x 35 x 35
        # output N x 256 x 17 x 17
        self.branch_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # input N x 256 x 35 x 35
        # output N x 384 x 17 x 17
        self.branch_2 = ConvBNRelu(in_channels=in_channels, out_channels=n, kernel_size=3, stride=2)
        self.branch_3 = nn.Sequential(
            # input N x 256 x 35 x 35
            ConvBNRelu(in_channels=in_channels, out_channels=k, kernel_size=1),
            # input N x 192 x 35 x 35
            ConvBNRelu(in_channels=k, out_channels=l, kernel_size=3, padding=1),
            # input N x 192 x 35 x 35
            ConvBNRelu(in_channels=l, out_channels=m, kernel_size=3, stride=2)
            # output N x 256 x 17 x 17
        )

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        out = torch.cat((x2, x3, x1), 1)
        return out


class InceptionResNetB(nn.Module):
    def __init__(self, in_channels=896, out_channels=896, scale=1.0):
        super(InceptionResNetB, self).__init__()
        self.scale = scale
        # input N x 896 x 17 x 17
        # output N x 128 x 17 x 17
        self.branch_1 = ConvBNRelu(in_channels=in_channels, out_channels=128, kernel_size=1)
        self.branch_2 = nn.Sequential(
            # input N x 896 x 17 x 17
            ConvBNRelu(in_channels=in_channels, out_channels=128, kernel_size=1),
            # input N x 128 x 17 x 17
            ConvBNRelu(in_channels=128, out_channels=128, kernel_size=(1, 7), padding=(0, 3)),
            # input N x 128 x 17 x 17
            ConvBNRelu(in_channels=128, out_channels=128, kernel_size=(7, 1), padding=(3, 0))
            # output N x 128 x 17 x 17
        )
        # input N x 256 x 17 x 17
        # output N x 896 x 17 x 17
        self.conv_branch = nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        branch_all = torch.cat((x1, x2), 1)
        out = self.conv_branch(branch_all)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class ReductionB(nn.Module):
    def __init__(self, in_channels=896):
        super(ReductionB, self).__init__()
        # input N x 896 x 17 x 17
        # output N x 896 x 8 x 8
        self.branch_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.branch_2 = nn.Sequential(
            # input N x 896 x 17 x 17
            ConvBNRelu(in_channels=in_channels, out_channels=256, kernel_size=1),
            # input N x 256 x 17 x 17
            ConvBNRelu(in_channels=256, out_channels=384, kernel_size=3, stride=2)
            # output N x 384 x 8 x 8
        )
        self.branch_3 = nn.Sequential(
            # input N x 896 x 17 x 17
            ConvBNRelu(in_channels=in_channels, out_channels=256, kernel_size=1),
            # input N x 256 x 17 x 17
            ConvBNRelu(in_channels=256, out_channels=256, kernel_size=3, stride=2)
            # output N x 256 x 8 x 8
        )
        self.branch_4 = nn.Sequential(
            # input N x 896 x 17 x 17
            ConvBNRelu(in_channels=in_channels, out_channels=256, kernel_size=1),
            # input N x 256 x 17 x 17
            ConvBNRelu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            # input N x 256 x 17 x 17
            ConvBNRelu(in_channels=256, out_channels=256, kernel_size=3, stride=2)
            # output N x 256 x 8 x 8
        )

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        x4 = self.branch_4(x)
        out = torch.cat((x2, x3, x4, x1), 1)
        return out


class InceptionResNetC(nn.Module):
    def __init__(self, in_channels=1792, out_channels=1792, scale=1.0, noReLU=False):
        super(InceptionResNetC, self).__init__()
        self.scale = scale
        self.noReLu = noReLU
        # input N x 1792 x 8 x 8
        # output N x 192 x 8 x 8
        self.branch_1 = ConvBNRelu(in_channels=in_channels, out_channels=192, kernel_size=1)
        self.branch_2 = nn.Sequential(
            # input N x 1792 x 8 x 8
            ConvBNRelu(in_channels=in_channels, out_channels=192, kernel_size=1),
            # input N x 192 x 8 x 8
            ConvBNRelu(in_channels=192, out_channels=192, kernel_size=(1, 3), padding=(0, 1)),
            # input N x 192 x 8 x 8
            ConvBNRelu(in_channels=192, out_channels=192, kernel_size=(3, 1), padding=(1, 0))
            # output N x 192 x 8 x 8
        )
        # input N x 384 x 8 x 8
        # output N x 1792 x 8 x 8
        self.conv_branch = nn.Conv2d(in_channels=384, out_channels=out_channels, kernel_size=1)
        if not self.noReLu:
            self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x1_cat_x2 = torch.cat((x1, x2), 1)
        out = self.conv_branch(x1_cat_x2)
        out = out * self.scale + x
        if not self.noReLu:
            out = self.relu(out)
        return out


class InceptionResNetV1(nn.Module):
    def __init__(self, out_channels=512, pretrained=None, dropout_prob=0.6, classify=None, num_classes=None, device=None):
        super(InceptionResNetV1, self).__init__()
        # in 3 out 256
        self.stem = Stem(in_channels=3, out_channels=256)
        # in 256 out 256
        self.inception_resnet_A_5 = nn.Sequential(
            InceptionResNetA(scale=0.17),
            InceptionResNetA(scale=0.17),
            InceptionResNetA(scale=0.17),
            InceptionResNetA(scale=0.17),
            InceptionResNetA(scale=0.17)
        )
        # in 256 out 896
        self.reduction_A = ReductionA(in_channels=256, k=192, l=192, m=256, n=384)
        # in 896 out 896
        self.inception_resnet_B_10 = nn.Sequential(
            InceptionResNetB(scale=0.10),
            InceptionResNetB(scale=0.10),
            InceptionResNetB(scale=0.10),
            InceptionResNetB(scale=0.10),
            InceptionResNetB(scale=0.10),
            InceptionResNetB(scale=0.10),
            InceptionResNetB(scale=0.10),
            InceptionResNetB(scale=0.10),
            InceptionResNetB(scale=0.10),
            InceptionResNetB(scale=0.10)
        )
        # in 896 out 1792
        self.reduction_B = ReductionB(in_channels=896)
        # in 1792 out 1792
        self.inception_resnet_C_5 = nn.Sequential(
            InceptionResNetC(scale=0.20),
            InceptionResNetC(scale=0.20),
            InceptionResNetC(scale=0.20),
            InceptionResNetC(scale=0.20),
            InceptionResNetC(scale=0.20)
        )
        self.inception_resnet_C_6 = InceptionResNetC(noReLU=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.last_linear = nn.Linear(in_features=1792, out_features=out_channels, bias=False)
        self.last_bn = nn.BatchNorm1d(out_channels, eps=0.001)

        self.classify = classify
        if pretrained is not None:
            if pretrained == 'vggface2':
                tmp_classes = 8631
            elif pretrained == 'casia-webface':
                tmp_classes = 10575
            elif pretrained == 'model':
                tmp_classes = 3
            else:
                raise Exception('No such model')
            self.logits = nn.Linear(512, tmp_classes)
            self.load_state_dict(
                torch.load(f'model/model_path/{pretrained}.pt')
            )

        if self.classify and self.num_classes is not None:
            self.logits = nn.Linear(512, num_classes)

        self.device = torch.device('cpu')
        if device is not None:
            self.device = device
            self.to(device)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_resnet_A_5(x)
        x = self.reduction_A(x)
        x = self.inception_resnet_B_10(x)
        x = self.reduction_B(x)
        x = self.inception_resnet_C_5(x)
        x = self.inception_resnet_C_6(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.size()[0], -1))
        x = self.last_bn(x)
        if self.classify:
            x = self.logits(x)
        else:
            x = F.normalize(x, p=2, dim=1)
        return x