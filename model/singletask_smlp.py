import torch
from torch import nn
from torchvision.models import resnet50



class sMLPBlock(nn.Module):
    def __init__(self, W, H, channels):
        super().__init__()
        assert W == H
        self.channels = channels
        self.activation = nn.GELU()
        self.BN = nn.BatchNorm2d(channels)
        self.proj_h = nn.Conv2d(H, H, (1, 1))
        self.proh_w = nn.Conv2d(W, W, (1, 1))
        self.fuse = nn.Conv2d(channels*3, channels, (1,1), (1,1), bias=False)

    def forward(self, x):
        x1 = self.activation(self.BN(x))
        x_h = self.proj_h(x1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x_w = self.proh_w(x1.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x2 = self.fuse(torch.cat([x, x_h, x_w], dim=1))
        x = x + x2
        return x


class smlp_resnet(nn.Module):
    def __init__(self, n_classes=2):
        super(smlp_resnet, self).__init__()
        self.resnet = resnet50(pretrained=True)
        
        self.resnet.fc =nn.Sequential(nn.Flatten(),nn.Linear(2048, 128),nn.Linear(128,n_classes), nn.LogSoftmax(dim=1))
        self.resnet = nn.Sequential(*list(self.resnet.children())[:])
        self.encode = self.resnet[:4]
        self.layer1 = self.resnet[4]
        self.layer2 = self.resnet[5]
        self.layer3 = self.resnet[6]
        self.layer4 = self.resnet[7]
        self.avgpool = self.resnet[8]
        self.fc = self.resnet[9]

        self.smlp_block1 = sMLPBlock(W=56, H=56, channels=64)
        self.smlp_block2 = sMLPBlock(W=56, H=56, channels=256)
        self.smlp_block3 = sMLPBlock(W=28, H=28, channels=512)
        self.smlp_block4 = sMLPBlock(W=14, H=14, channels=1024)
        self.smlp_block5 = sMLPBlock(W=7, H=7, channels=2048)
        
    def forward(self, x):
        x = self.encode(x)
        x = self.smlp_block1(x)
        x = self.layer1(x)
        x = self.smlp_block2(x)
        x = self.layer2(x)
        x = self.smlp_block3(x)
        x = self.layer3(x)
        x = self.smlp_block4(x)
        x = self.layer4(x)
        x = self.smlp_block5(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x
    


model = smlp_resnet(n_classes=2)
model.cuda()
if __name__ == '__main__':
    print(model)
