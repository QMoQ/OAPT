
import sys
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY
import torch.nn.functional as F
from oapt.archs.swiniroffsetdense_hw_arch import OAPT_gt
# from swiniroffsetdense_hw_arch import OAPT_gt



class ResBlock_small(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock_small, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False,groups=outchannel),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False,groups=outchannel),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            #shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            
    def forward(self, x):
        out = self.left(x)
        #将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out
        
class ResNet18_small(nn.Module):
    def __init__(self, inch=1, num_classes=4):
        super(ResNet18_small, self).__init__()
        self.nc = num_classes
        num_classes = (num_classes+1)*(num_classes+1)
        self.inch = inch
        self.conv1 = nn.Sequential(
            nn.Conv2d(inch, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.inchannel = 64
        self.layer1 = self.make_layer(ResBlock_small, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock_small, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock_small, 256, 2, stride=2)        
        self.layer4 = self.make_layer(ResBlock_small, 512, 2, stride=2)        
        self.fc = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(num_classes, 2)
        self.sig = nn.Sigmoid()

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        s1 = x.shape[1]
        if s1==3 and self.inch==1: # This is for the RGB input but only need to predict on Y channel
            x = 24.966*x[:,0,:,:]+128.553*x[:,1,:,:]+65.481*x[:,2,:,:]+16.0
            x = (x/255.).unsqueeze(1)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.sig(self.fc2(out))*self.nc
        # out = self.fc2(out)
        return out




@ARCH_REGISTRY.register()
class OAPT(nn.Module):
    def __init__(self, 
                predictor_type='resnet18',# 预测模型
                img_size=64, patch_size=1, in_chans=3,
                embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                window_size=7, DenseFirstPositions=True,mode='interval',
                mlp_ratio=4., qkv_bias=True, qk_scale=None,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                use_checkpoint=False, upscale=2, img_range=1., upsampler='', 
                resi_connection='1conv', exp=False, conv=False,
                offset_range=4,
                **kwargs):
        super(OAPT, self).__init__()

        self.prediction = ResNet18_small(inch=in_chans, num_classes=offset_range)
        self.restoration = OAPT_gt(img_size=img_size, patch_size=patch_size, 
                                                            in_chans=in_chans,
                                                            embed_dim=embed_dim, 
                                                            depths=depths, 
                                                            num_heads=num_heads,
                                                            window_size=window_size, 
                                                            DenseFirstPositions=DenseFirstPositions,
                                                            mode=mode,
                                                            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                                            drop_path_rate=drop_path_rate,
                                                            norm_layer=norm_layer, ape=ape, patch_norm=patch_norm,
                                                            use_checkpoint=use_checkpoint, 
                                                            upscale=upscale, 
                                                            img_range=img_range, 
                                                            upsampler=upsampler, 
                                                            resi_connection=resi_connection,**kwargs)

    def forward(self, x):
        x_chop = x[:,:,:self.left_top_size,:self.left_top_size]
        offset_pred = self.prediction(x_chop) # 转化offset的one-hot模式为[oh, ow]
        output = self.restoration(x, offset_pred)
        return offset_pred, output
    
    def test_forward(self, x, offset=0, test=False):
        if not test:
            x_chop = x[:,:,:self.left_top_size,:self.left_top_size]
            offset_pred = self.prediction(x_chop)
            output = self.restoration(x, offset_pred)
        else:
            offset_pred = offset
            output = self.restoration(x, offset)
        return offset_pred, output



if __name__ == '__main__':

    height = 44 #160
    width = 44 #160
    model = OAPT(
        predictor_type="resnet18_small",
        upscale=1,
        in_chans=1,
        img_size=224,
        window_size=7,
        DenseFirstPositions=True,
        mode='non-interval',
        img_range=255.,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='',
        resi_connection='1conv',
        offset_range=7).prediction

    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model,(1, height, width),as_strings=True,print_per_layer_stat=False)
    print('MACs:  ' + macs)
    print('Params: ' + params)