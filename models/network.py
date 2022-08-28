import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
import math


class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels, convNum, normLayer=None):
        super(ConvBlock, self).__init__()
        self.inConv = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        layers = []
        for _ in range(convNum - 1):
            layers.append(nn.Conv2d(outChannels, outChannels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        if not (normLayer is None):
            layers.append(normLayer(outChannels))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.inConv(x)
        x = self.conv(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels, normLayer=None):
        super(ResidualBlock, self).__init__()
        layers = []
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        layers.append(spectral_norm(nn.Conv2d(channels, channels, kernel_size=3, padding=1)))
        if not (normLayer is None):
            layers.append(normLayer(channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        if not (normLayer is None):
            layers.append(normLayer(channels))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        residual = self.conv(x)
        return F.relu(x + residual, inplace=True)


class ResidualBlockSN(nn.Module):
    def __init__(self, channels, normLayer=None):
        super(ResidualBlockSN, self).__init__()
        layers = []
        layers.append(spectral_norm(nn.Conv2d(channels, channels, kernel_size=3, padding=1)))
        layers.append(nn.LeakyReLU(0.2, True))
        layers.append(spectral_norm(nn.Conv2d(channels, channels, kernel_size=3, padding=1)))
        if not (normLayer is None):
            layers.append(normLayer(channels))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        residual = self.conv(x)
        return F.leaky_relu(x + residual, 2e-1, inplace=True)


class DownsampleBlock(nn.Module):
    def __init__(self, inChannels, outChannels, convNum=2, normLayer=None):
        super(DownsampleBlock, self).__init__()
        layers = []
        layers.append(nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1, stride=2))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(convNum - 1):
            layers.append(nn.Conv2d(outChannels, outChannels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        if not (normLayer is None):
            layers.append(normLayer(outChannels))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class UpsampleBlock(nn.Module):
    def __init__(self, inChannels, outChannels, convNum=2, normLayer=None):
        super(UpsampleBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1, stride=1)
        self.combine = nn.Conv2d(2 * outChannels, outChannels, kernel_size=3, padding=1)
        layers = []
        for _ in range(convNum - 1):
            layers.append(nn.Conv2d(outChannels, outChannels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        if not (normLayer is None):
            layers.append(normLayer(outChannels))
        self.conv2 = nn.Sequential(*layers)

    def forward(self, x, x0):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.combine(torch.cat((x, x0), 1))
        x = F.relu(x)
        return self.conv2(x)


class UpsampleBlockSN(nn.Module):
    def __init__(self, inChannels, outChannels, convNum=2, normLayer=None):
        super(UpsampleBlockSN, self).__init__()
        self.conv1 = spectral_norm(nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=1, padding=1))
        self.shortcut = spectral_norm(nn.Conv2d(outChannels, outChannels, kernel_size=3, stride=1, padding=1))
        layers = []
        for _ in range(convNum - 1):
            layers.append(spectral_norm(nn.Conv2d(outChannels, outChannels, kernel_size=3, padding=1)))
            layers.append(nn.LeakyReLU(0.2, True))
        if not (normLayer is None):
            layers.append(normLayer(outChannels))
        self.conv2 = nn.Sequential(*layers)

    def forward(self, x, x0):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = x + self.shortcut(x0)
        x = F.leaky_relu(x, 2e-1)
        return self.conv2(x)


class HourGlass2(nn.Module):
    def __init__(self, inChannel=3, outChannel=1, resNum=3, normLayer=None):
        super(HourGlass2, self).__init__()
        self.inConv = ConvBlock(inChannel, 64, convNum=2, normLayer=normLayer)
        self.down1 = DownsampleBlock(64, 128, convNum=2, normLayer=normLayer)
        self.down2 = DownsampleBlock(128, 256, convNum=2, normLayer=normLayer)
        self.residual = nn.Sequential(*[ResidualBlock(256) for _ in range(resNum)])
        self.up2 = UpsampleBlock(256, 128, convNum=3, normLayer=normLayer)
        self.up1 = UpsampleBlock(128, 64, convNum=3, normLayer=normLayer)
        self.outConv = nn.Conv2d(64, outChannel, kernel_size=3, padding=1)

    def forward(self, x):
        f1 = self.inConv(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        r3 = self.residual(f3)
        r2 = self.up2(r3, f2)
        r1 = self.up1(r2, f1)
        y = self.outConv(r1)
        return y


class ColorProbNet(nn.Module):
    def __init__(self, inChannel=1, outChannel=2, with_SA=False):
        super(ColorProbNet, self).__init__()
        BNFunc = nn.BatchNorm2d
        # conv1: 256
        conv1_2 = [spectral_norm(nn.Conv2d(inChannel, 64, 3, stride=1, padding=1)), nn.LeakyReLU(0.2, True),]
        conv1_2 += [spectral_norm(nn.Conv2d(64, 64, 3, stride=1, padding=1)), nn.LeakyReLU(0.2, True),]
        conv1_2 += [BNFunc(64, affine=True)]
        # conv2: 128
        conv2_3 = [spectral_norm(nn.Conv2d(64, 128, 3, stride=2, padding=1)), nn.LeakyReLU(0.2, True),]
        conv2_3 += [spectral_norm(nn.Conv2d(128, 128, 3, stride=1, padding=1)), nn.LeakyReLU(0.2, True),] 
        conv2_3 += [spectral_norm(nn.Conv2d(128, 128, 3, stride=1, padding=1)), nn.LeakyReLU(0.2, True),] 
        conv2_3 += [BNFunc(128, affine=True)]
        # conv3: 64
        conv3_3 = [spectral_norm(nn.Conv2d(128, 256, 3, stride=2, padding=1)), nn.LeakyReLU(0.2, True),]
        conv3_3 += [spectral_norm(nn.Conv2d(256, 256, 3, stride=1, padding=1)), nn.LeakyReLU(0.2, True),]
        conv3_3 += [spectral_norm(nn.Conv2d(256, 256, 3, stride=1, padding=1)), nn.LeakyReLU(0.2, True),]  
        conv3_3 += [BNFunc(256, affine=True)]
        # conv4: 32
        conv4_3 = [spectral_norm(nn.Conv2d(256, 512, 3, stride=2, padding=1)), nn.LeakyReLU(0.2, True),]
        conv4_3 += [spectral_norm(nn.Conv2d(512, 512, 3, stride=1, padding=1)), nn.LeakyReLU(0.2, True),] 
        conv4_3 += [spectral_norm(nn.Conv2d(512, 512, 3, stride=1, padding=1)), nn.LeakyReLU(0.2, True),]
        conv4_3 += [BNFunc(512, affine=True)]
        # conv5: 32
        conv5_3 = [spectral_norm(nn.Conv2d(512, 512, 3, stride=1, padding=1)), nn.LeakyReLU(0.2, True),]
        conv5_3 += [spectral_norm(nn.Conv2d(512, 512, 3, stride=1, padding=1)), nn.LeakyReLU(0.2, True),]
        conv5_3 += [spectral_norm(nn.Conv2d(512, 512, 3, stride=1, padding=1)), nn.LeakyReLU(0.2, True),]
        conv5_3 += [BNFunc(512, affine=True)]
        # conv6: 32
        conv6_3 = [spectral_norm(nn.Conv2d(512, 512, 3, stride=1, padding=1)), nn.LeakyReLU(0.2, True),] 
        conv6_3 += [spectral_norm(nn.Conv2d(512, 512, 3, stride=1, padding=1)), nn.LeakyReLU(0.2, True),]
        conv6_3 += [spectral_norm(nn.Conv2d(512, 512, 3, stride=1, padding=1)), nn.LeakyReLU(0.2, True),]
        conv6_3 += [BNFunc(512, affine=True),]
        if with_SA:
            conv6_3 += [Self_Attn(512)]
        # conv7: 32
        conv7_3 = [spectral_norm(nn.Conv2d(512, 512, 3, stride=1, padding=1)), nn.LeakyReLU(0.2, True),]
        conv7_3 += [spectral_norm(nn.Conv2d(512, 512, 3, stride=1, padding=1)), nn.LeakyReLU(0.2, True),]  
        conv7_3 += [spectral_norm(nn.Conv2d(512, 512, 3, stride=1, padding=1)), nn.LeakyReLU(0.2, True),]
        conv7_3 += [BNFunc(512, affine=True)]
        # conv8: 64
        conv8up = [nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(512, 256, 3, stride=1, padding=1),]
        conv3short8 = [nn.Conv2d(256, 256, 3, stride=1, padding=1),]
        conv8_3 = [nn.ReLU(True),]
        conv8_3 += [nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.ReLU(True),]
        conv8_3 += [nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.ReLU(True),]
        conv8_3 += [BNFunc(256, affine=True),]
        # conv9: 128
        conv9up = [nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(256, 128, 3, stride=1, padding=1),]
        conv9_2 = [nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.ReLU(True),]
        conv9_2 += [BNFunc(128, affine=True)]
        # conv10: 64
        conv10up = [nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(128, 64, 3, stride=1, padding=1),]
        conv10_2 = [nn.ReLU(True),]
        conv10_2 += [nn.Conv2d(64, outChannel, 3, stride=1, padding=1), nn.ReLU(True),]
        
        self.conv1_2 = nn.Sequential(*conv1_2)
        self.conv2_3 = nn.Sequential(*conv2_3)
        self.conv3_3 = nn.Sequential(*conv3_3)
        self.conv4_3 = nn.Sequential(*conv4_3)
        self.conv5_3 = nn.Sequential(*conv5_3)
        self.conv6_3 = nn.Sequential(*conv6_3)
        self.conv7_3 = nn.Sequential(*conv7_3)
        self.conv8up = nn.Sequential(*conv8up)
        self.conv3short8 = nn.Sequential(*conv3short8)
        self.conv8_3 = nn.Sequential(*conv8_3)
        self.conv9up = nn.Sequential(*conv9up)
        self.conv9_2 = nn.Sequential(*conv9_2)
        self.conv10up = nn.Sequential(*conv10up)
        self.conv10_2 = nn.Sequential(*conv10_2)
        # claffificaton output
        #self.model_class = nn.Sequential(*[nn.Conv2d(256, 313, kernel_size=1, padding=0, stride=1),])

    def forward(self, input_grays):
        f1_2 = self.conv1_2(input_grays)
        f2_3 = self.conv2_3(f1_2)
        f3_3 = self.conv3_3(f2_3)
        f4_3 = self.conv4_3(f3_3)
        f5_3 = self.conv5_3(f4_3)
        f6_3 = self.conv6_3(f5_3)
        f7_3 = self.conv7_3(f6_3)
        f8_up = self.conv8up(f7_3) + self.conv3short8(f3_3)
        f8_3 = self.conv8_3(f8_up)
        f9_up = self.conv9up(f8_3)
        f9_2 = self.conv9_2(f9_up)
        f10_up = self.conv10up(f9_2)
        f10_2 = self.conv10_2(f10_up)
        out_feats = f10_2
        #out_probs = self.model_class(f8_3)
        return out_feats



def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1)
        )


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1)
    )

class SpixelNet(nn.Module):
    def __init__(self, inChannel=3, outChannel=9, batchNorm=True):
        super(SpixelNet,self).__init__()
        self.batchNorm = batchNorm
        self.conv0a = conv(self.batchNorm, inChannel, 16, kernel_size=3)
        self.conv0b = conv(self.batchNorm, 16, 16, kernel_size=3)
        self.conv1a = conv(self.batchNorm, 16, 32, kernel_size=3, stride=2)
        self.conv1b = conv(self.batchNorm, 32, 32, kernel_size=3)
        self.conv2a = conv(self.batchNorm, 32, 64, kernel_size=3, stride=2)
        self.conv2b = conv(self.batchNorm, 64, 64, kernel_size=3)
        self.conv3a = conv(self.batchNorm, 64, 128, kernel_size=3, stride=2)
        self.conv3b = conv(self.batchNorm, 128, 128, kernel_size=3)
        self.conv4a = conv(self.batchNorm, 128, 256, kernel_size=3, stride=2)
        self.conv4b = conv(self.batchNorm, 256, 256, kernel_size=3)
        self.deconv3 = deconv(256, 128)
        self.conv3_1 = conv(self.batchNorm, 256, 128)
        self.deconv2 = deconv(128, 64)
        self.conv2_1 = conv(self.batchNorm, 128, 64)
        self.deconv1 = deconv(64, 32)
        self.conv1_1 = conv(self.batchNorm, 64, 32)
        self.deconv0 = deconv(32, 16)
        self.conv0_1 = conv(self.batchNorm, 32, 16)
        self.pred_mask0 = nn.Conv2d(16, outChannel, kernel_size=3, stride=1, padding=1, bias=True)
        self.softmax = nn.Softmax(1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        out1 = self.conv0b(self.conv0a(x))    #5*5
        out2 = self.conv1b(self.conv1a(out1)) #11*11
        out3 = self.conv2b(self.conv2a(out2)) #23*23
        out4 = self.conv3b(self.conv3a(out3)) #47*47
        out5 = self.conv4b(self.conv4a(out4)) #95*95
        out_deconv3 = self.deconv3(out5)
        concat3 = torch.cat((out4, out_deconv3), 1)
        out_conv3_1 = self.conv3_1(concat3)
        out_deconv2 = self.deconv2(out_conv3_1)
        concat2 = torch.cat((out3, out_deconv2), 1)
        out_conv2_1 = self.conv2_1(concat2)
        out_deconv1 = self.deconv1(out_conv2_1)
        concat1 = torch.cat((out2, out_deconv1), 1)
        out_conv1_1 = self.conv1_1(concat1)
        out_deconv0 = self.deconv0(out_conv1_1)
        concat0 = torch.cat((out1, out_deconv0), 1)
        out_conv0_1 = self.conv0_1(concat0)
        mask0 = self.pred_mask0(out_conv0_1)
        prob0 = self.softmax(mask0)
        return prob0



## VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False, local_pretrained_path='checkpoints/vgg19.pth'):
        super().__init__()
        #vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        model = torchvision.models.vgg19()
        model.load_state_dict(torch.load(local_pretrained_path))
        vgg_pretrained_features = model.features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out