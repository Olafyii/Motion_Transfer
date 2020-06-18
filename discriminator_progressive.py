'''
    progressive version of discriminator.py
    basically same as discriminator.py, except more layers in U-Net
'''
import torch
import torch.nn as nn

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

class Discriminator(nn.Module):
    def __init__(self, params, in_channels=3, out_channels=1, bias=True, norm='batch', sigmoid=True):
        super(Discriminator, self).__init__()
        self.sigmoid = sigmoid

        # 70x70 discriminator
        self.disc1 = EncoderBlock(in_channels, 64, bias=bias, do_norm=False, do_activation=False)
        self.disc15 = EncoderBlock(64, 64, bias=bias, norm=norm)
        self.disc2 = EncoderBlock(64+params['n_joints'], 128, bias=bias, norm=norm)
        self.disc3 = EncoderBlock(128, 256, bias=bias, norm=norm)
        self.disc4 = EncoderBlock(256, 512, bias=bias, norm=norm, stride=1)
        self.disc5 = EncoderBlock(512, out_channels, bias=bias, stride=1, do_norm=False)

    def forward(self, x, pose):
        d0 = self.disc1(x)
        d15 = self.disc15(d0)
        d1 = torch.cat((d15, pose), dim=1)
        d2 = self.disc2(d1)
        d3 = self.disc3(d2)
        d4 = self.disc4(d3)
        d5 = self.disc5(d4)
        if self.sigmoid:
            final = nn.Sigmoid()(d5)
        else:
            final = d5
        return final

        
if __name__ == '__main__':
    a = torch.randn((2, 3, 512, 512))
    b = torch.randn((2, 14, 128, 128))
    from param import get_general_params
    params = get_general_params()
    model = Discriminator(params)
    out = model(a, b)
    weight = torch.ones_like(out)
    print(weight.size())
    loss = nn.BCELoss(weight=torch.ones_like(out))(out, torch.zeros_like(out))
    print(out.size())
    print(loss)