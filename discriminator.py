'''
    定义了全图的判别器Discriminator和人脸判别器PatchFace
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, dilation=1, groups=1, bias=False,
                 do_norm=True, norm = 'batch', do_activation = True): # bias default is True in Conv2d
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.leakyRelu = nn.LeakyReLU(0.2, True)
        self.do_norm = do_norm
        self.do_activation = do_activation
        if do_norm:
            if norm == 'batch':
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm == 'instance':
                self.norm = nn.InstanceNorm2d(out_channels)
            elif norm == 'none':
                self.do_norm = False
            else:
                raise NotImplementedError("norm error")

    def forward(self, x):
        if self.do_activation:
            x = self.leakyRelu(x)

        x = self.conv(x)

        if self.do_norm:
            x = self.norm(x)

        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False,
                 do_norm=True, norm = 'batch',do_activation = True, dropout_prob=0.0):
        super(DecoderBlock, self).__init__()

        self.convT = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.relu = nn.ReLU()
        self.dropout_prob = dropout_prob
        self.drop = nn.Dropout2d(dropout_prob)
        self.do_norm = do_norm
        self.do_activation = do_activation
        if do_norm:
            if norm == 'batch':
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm == 'instance':
                self.norm = nn.InstanceNorm2d(out_channels)
            elif norm == 'none':
                self.do_norm = False
            else:
                raise NotImplementedError("norm error")

    def forward(self, x):
        if self.do_activation:
            x = self.relu(x)

        x = self.convT(x)

        if self.do_norm:
           x = self.norm(x)

        if self.dropout_prob != 0:
            x= self.drop(x)

        return x

class Discriminator(nn.Module):  # 70*70 PatchGAN
    '''
        输入有两个，一个是3通道的rgb图，一个是14通道的人体关键点，一个通道一个点。
    '''
    def __init__(self, params, in_channels=3, out_channels=1, bias=True, norm='batch', sigmoid=True):
        super(Discriminator, self).__init__()
        self.sigmoid = sigmoid

        # 70x70 discriminator
        self.disc1 = EncoderBlock(in_channels, 64, bias=bias, do_norm=False, do_activation=False)
        self.disc2 = EncoderBlock(64+params['n_joints'], 128, bias=bias, norm=norm)
        self.disc3 = EncoderBlock(128, 256, bias=bias, norm=norm)
        self.disc4 = EncoderBlock(256, 512, bias=bias, norm=norm, stride=1)
        self.disc5 = EncoderBlock(512, out_channels, bias=bias, stride=1, do_norm=False)

    def forward(self, x, pose):
        # 70*70
        d0 = self.disc1(x)
        d1 = torch.cat((d0, pose), dim=1)
        d2 = self.disc2(d1)
        d3 = self.disc3(d2)
        d4 = self.disc4(d3)
        d5 = self.disc5(d4)
        # print('size of d0', d0.size())
        # print('size of d1', d1.size())
        # print('size of d2', d2.size())
        # print('size of d3', d3.size())
        # print('size of d4', d4.size())
        # print('size of d5', d5.size())
        if self.sigmoid:
            final = nn.Sigmoid()(d5)
        else:
            final = d5
        return final
    
class FaceDisc(nn.Module):  # deprecated
    def __init__(self, in_channels=3, out_channels=1, bias=True, norm='batch', sigmoid=True):
        super(FaceDisc, self).__init__()
        self.sigmoid = sigmoid

        # 70x70 discriminator
        self.disc1 = EncoderBlock(in_channels, 64, bias=bias, do_norm=False, do_activation=False)
        self.disc2 = EncoderBlock(64, 128, bias=bias, norm=norm)
        self.disc3 = EncoderBlock(128, 256, bias=bias, stride=1, do_norm=False)
        self.linear1 = nn.Linear(256*63*63, 256)
        self.linear2 = nn.Linear(256, 1)

    def forward(self, x):
        d1 = self.disc1(x)
        d2 = self.disc2(d1)
        d3 = self.disc3(d2)
        d3 = F.leaky_relu(d3, 0.2)
        # print(d3.size())
        d3 = d3.view(d3.size(0), -1)
        l1 = self.linear1(d3)
        l1 = F.leaky_relu(l1, 0.2)
        l2 = self.linear2(l1)
        l2 = F.sigmoid(l2)

        return l2

class PatchFace(nn.Module):  # human face PatchGAN
    '''
        输入为4通道，其中3通道为人脸rgb图，第四个通道是人脸关键点heatmap。共68个关键点，在一个通道里。
    '''
    def __init__(self, in_channels=4, out_channels=1, bias=True, norm='batch', sigmoid=True):
        super(PatchFace, self).__init__()
        self.sigmoid = sigmoid

        # 70x70 discriminator
        self.disc1 = EncoderBlock(in_channels, 64, bias=bias, do_norm=False, do_activation=False)
        self.disc2 = EncoderBlock(64, 128, bias=bias, norm=norm)
        self.disc3 = EncoderBlock(128, 256, bias=bias, norm=norm)
        self.disc4 = EncoderBlock(256, 512, bias=bias, norm=norm, stride=1)
        self.disc5 = EncoderBlock(512, out_channels, bias=bias, stride=1, do_norm=False)

    def forward(self, x):
        # 70*70
        d0 = self.disc1(x)
        # d1 = torch.cat((d0, pose), dim=1)
        d1 = d0
        d2 = self.disc2(d1)
        d3 = self.disc3(d2)
        d4 = self.disc4(d3)
        d5 = self.disc5(d4)
        # print('size of d0', d0.size())
        # print('size of d1', d1.size())
        # print('size of d2', d2.size())
        # print('size of d3', d3.size())
        # print('size of d4', d4.size())
        # print('size of d5', d5.size())
        if self.sigmoid:
            final = nn.Sigmoid()(d5)
        else:
            final = d5
        return final
        
if __name__ == '__main__':
    # img = torch.randn((1, 3, 256, 256))
    # pose = torch.randn((1, 14, 128, 128))
    # from param import get_general_params
    # params = get_general_params()
    # model = Discriminator(params)
    # out = model(img, pose)
    # loss = nn.BCELoss()(out, torch.zeros_like(out))
    # print(out.size())
    # print(loss)
    i = torch.randn((2, 3, 256, 256))
    model = FaceDisc()
    out = model(i)
    print(out)