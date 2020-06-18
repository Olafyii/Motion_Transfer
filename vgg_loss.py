'''
    Defines vgg loss function.
'''
import torch
import torchvision
import torch.nn as nn

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        
        input = input*0.5+0.5
        target = target*0.5+0.5

        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss


class L1MaskLoss(nn.Module):
    def __init__(self):
        super(L1MaskLoss, self).__init__()
    def forward(self, input, target, mask):
        mask = torch.cat((mask, mask, mask), dim=1)
        mask = nn.Threshold(0.01, 0)(mask).bool()
        input = torch.masked_select(input, mask)
        target = torch.masked_select(target, mask)

        return nn.L1Loss()(input, target)

if __name__ == '__main__':
    # a = torch.randn((16, 3, 256, 256))
    # b = torch.randn((16, 3, 256, 256))
    # model = VGGPerceptualLoss()
    # out = model(a, b)
    # print(model)
    # print(out)
    # print('out.relu1_2.size()', out.relu1_2.size())
    # print('out.relu2_2.size()', out.relu2_2.size())
    # print('out.relu3_3.size()', out.relu3_3.size())
    # print('out.relu4_3.size()', out.relu4_3.size())
    print(torchvision.models.vgg16().features)