import torch
import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel
from .model_zoo import MODEL_REGISTRY
from ..modules import _ConvBNReLU, SeparableConv2d, _A_ASPP, _FCNHead
from ..config import cfg
from IPython import embed
import math
from collections import OrderedDict
import numpy as np
__all__ = ['FANet']

def _resize_image(img, h, w):
    return F.interpolate(img, size=[h, w], mode='bilinear', align_corners=True)


@MODEL_REGISTRY.register(name='FANet')
class FANet(SegBaseModel):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(FANet, self).__init__()
        if self.backbone.startswith('mobilenet'):
            c1_channels = 24
            c4_channels = 320
        else:
            c1_channels = 256
            c4_channels = 2048
            c2_channels = 512
            c3_channels = 1024
        
        self.fusion_head = _Decoder(c1_channels, c2_channels, c3_channels, c4_channels, self.nclass)
        self.__setattr__('decoder', ['fusion_head'])

    def forward(self, x, show_middle=False, show_scores=False):
        size = x.size()[2:] 
        c1, c2, c3, c4 = self.encoder(x)
        if(show_middle):
            y_s, y_b, middle_before_aspp, middle_after_aspp, middle_fusion = self.fusion_head(c1, c2, c3, c4, size, show_middle)
        elif(show_scores):
            y_s, y_b, middle_scores = self.fusion_head(c1, c2, c3, c4, size, show_scores=show_scores)
        else:
            y_s, y_b = self.fusion_head(c1, c2, c3, c4, size)
        outputs_s = list()
        outputs_b = list()
        outputs_s.append(y_s)
        outputs_b.append(y_b)#.sigmoid())

        if(show_middle):
            return tuple(outputs_s), tuple(outputs_b), middle_before_aspp, middle_after_aspp, middle_fusion
        elif(show_scores):
            return tuple(outputs_s), tuple(outputs_b), middle_scores
        else:
            return tuple(outputs_s), tuple(outputs_b)
         

    def evaluate(self, image, show_middle=False, show_scores=False):
        """evaluating network with inputs and targets"""
        scales = cfg.TEST.SCALES
        batch, _, h, w = image.shape
        base_size = max(h, w)
        # scores = torch.zeros((batch, self.nclass, h, w)).to(image.device)
        scores = None
        scores_boundary = None
        for scale in scales:
            long_size = int(math.ceil(base_size * scale))
            if h > w:
                height = long_size
                width = int(1.0 * w * long_size / h + 0.5)
            else:
                width = long_size
                height = int(1.0 * h * long_size / w + 0.5)

            # resize image to current size
            cur_img = _resize_image(image, height, width)
            if(show_middle==False and show_scores==False):
                outputs, outputs_boundary = self.forward(cur_img)
            elif(show_middle==True and show_scores==False):
                outputs, outputs_boundary, middle_before_aspp, middle_after_aspp, middle_fusion = self.forward(cur_img, show_middle=True)
            elif(show_middle==False and show_scores==True):
                outputs, outputs_boundary, middle_scores = self.forward(cur_img, show_scores=True)
            #print(np.unique(outputs_boundary[0].cpu().numpy()))
            outputs = outputs[0][..., :height, :width]
            outputs_boundary = outputs_boundary[0][..., :height, :width]

            score = _resize_image(outputs, h, w)
            score_boundary = _resize_image(outputs_boundary, h, w)

            if scores is None:
                scores = score
                scores_boundary = score_boundary
            else:
                scores += score
                scores_boundary += score_boundary
        if(show_middle):
            return scores, scores_boundary, middle_before_aspp, middle_after_aspp, middle_fusion
        elif(show_scores):
            return scores, scores_boundary, middle_scores
        else:
            return scores, scores_boundary

class _Decoder(nn.Module):
    def __init__(self, c1_channels, c2_channels, c3_channels, c4_channels, nclass, norm_layer=nn.BatchNorm2d):
        super(_Decoder, self).__init__()   

        self.nclass = nclass
        self.daspp1 = _A_ASPP(c1_channels, 256)
        self.daspp2 = _A_ASPP(c2_channels, 256)
        self.daspp3 = _A_ASPP(c3_channels, 256)
        self.daspp4 = _A_ASPP(c4_channels, 256)

        self.activation_s = nn.ReLU()
        self.activation_b = nn.Sigmoid()

        #self.s_to_b_1 = nn.Sequential(OrderedDict([('conv', nn.Conv2d(256, 256, 1, bias=False)),
                                                #('bn', nn.BatchNorm2d(256)),
                                                #('relu', nn.ReLU())]))
        self.b_to_s_1 = SeparableConv2d(256, 256, 3, norm_layer=norm_layer, relu_first=False, sigmoid_last=True) #nn.Sequential(OrderedDict([('conv', nn.Conv2d(256, 256, 1, bias=False)),
                        #                        ('bn', nn.BatchNorm2d(256)),
                        #                        ('sigmoid', self.activation_b)]))
        #self.s_to_b_2 = nn.Sequential(OrderedDict([('conv', nn.Conv2d(256, 256, 1, bias=False)),
                                                #('bn', nn.BatchNorm2d(256)),
                                                #('relu', nn.ReLU())]))
        self.b_to_s_2 = SeparableConv2d(256, 256, 3, norm_layer=norm_layer, relu_first=False, sigmoid_last=True) #nn.Sequential(OrderedDict([('conv', nn.Conv2d(256, 256, 1, bias=False)),
                        #                        ('bn', nn.BatchNorm2d(256)),
                        #                        ('sigmoid', self.activation_b)]))
        #self.s_to_b_3 = nn.Sequential(OrderedDict([('conv', nn.Conv2d(256, 256, 1, bias=False)),
                                                #('bn', nn.BatchNorm2d(256)),
                                                #('relu', nn.ReLU())]))
        self.b_to_s_3 = SeparableConv2d(256, 256, 3, norm_layer=norm_layer, relu_first=False, sigmoid_last=True) #nn.Sequential(OrderedDict([('conv', nn.Conv2d(256, 256, 1, bias=False)),
                        #                        ('bn', nn.BatchNorm2d(256)),
                        #                        ('sigmoid', self.activation_b)]))
        #self.s_to_b_4 = nn.Sequential(OrderedDict([('conv', nn.Conv2d(256, 256, 1, bias=False)),
                                                #('bn', nn.BatchNorm2d(256)),
                                                #('relu', nn.ReLU())]))
        self.b_to_s_4 = SeparableConv2d(256, 256, 3, norm_layer=norm_layer, relu_first=False, sigmoid_last=True) #nn.Sequential(OrderedDict([('conv', nn.Conv2d(256, 256, 1, bias=False)),
                        #                        ('bn', nn.BatchNorm2d(256)),
                        #                        ('sigmoid', self.activation_b)]))

        self.conv_s_4 = SeparableConv2d(256, 256, 3, norm_layer=norm_layer, relu_first=False) #nn.Sequential(OrderedDict([('conv', nn.Conv2d(256, 256, 3, bias=False)),
                        #                        ('bn', nn.BatchNorm2d(256)),
                        #                        ('relu', nn.ReLU())]))
        self.conv_s_3 = SeparableConv2d(256, 256, 3, norm_layer=norm_layer, relu_first=False) #nn.Sequential(OrderedDict([('conv', nn.Conv2d(256, 256, 3, bias=False)),
                        #                        ('bn', nn.BatchNorm2d(256)),
                        #                        ('relu', nn.ReLU())]))
        self.conv_s_2 = SeparableConv2d(256, 256, 3, norm_layer=norm_layer, relu_first=False) #nn.Sequential(OrderedDict([('conv', nn.Conv2d(256, 256, 3, bias=False)),
                        #                        ('bn', nn.BatchNorm2d(256)),
                        #                        ('relu', nn.ReLU())]))
        self.conv_s_1 = SeparableConv2d(256, 256, 3, norm_layer=norm_layer, relu_first=False) #nn.Sequential(OrderedDict([('conv', nn.Conv2d(256, 256, 3, bias=False)),
                        #                        ('bn', nn.BatchNorm2d(256)),
                        #                        ('relu', nn.ReLU())]))

        self.conv_b_4 = SeparableConv2d(256, 256, 3, norm_layer=norm_layer, relu_first=False) #nn.Sequential(OrderedDict([('conv', nn.Conv2d(256, 256, 3, bias=False)),
                        #                        ('bn', nn.BatchNorm2d(256)),
                        #                        ('relu', nn.ReLU())]))
        self.conv_b_3 = SeparableConv2d(256, 256, 3, norm_layer=norm_layer, relu_first=False) #nn.Sequential(OrderedDict([('conv', nn.Conv2d(256, 256, 3, bias=False)),
                        #                        ('bn', nn.BatchNorm2d(256)),
                        #                        ('relu', nn.ReLU())]))
        self.conv_b_2 = SeparableConv2d(256, 256, 3, norm_layer=norm_layer, relu_first=False) #nn.Sequential(OrderedDict([('conv', nn.Conv2d(256, 256, 3, bias=False)),
                        #                        ('bn', nn.BatchNorm2d(256)),
                        #                        ('relu', nn.ReLU())]))
        self.conv_b_1 = SeparableConv2d(256, 256, 3, norm_layer=norm_layer, relu_first=False) #nn.Sequential(OrderedDict([('conv', nn.Conv2d(256, 256, 3, bias=False)),
                        #                        ('bn', nn.BatchNorm2d(256)),
                        #                        ('relu', nn.ReLU())]))
        self.block_s = nn.Sequential(
            SeparableConv2d(256, 256, 3, norm_layer=norm_layer, relu_first=False),
            SeparableConv2d(256, 256, norm_layer=norm_layer, relu_first=False),
            nn.Conv2d(256, self.nclass, 1))
        self.block_b = nn.Sequential(
            SeparableConv2d(256, 256, 3, norm_layer=norm_layer, relu_first=False),
            SeparableConv2d(256, 256, 3, norm_layer=norm_layer, relu_first=False),
            nn.Conv2d(256, 1, 1))

    def forward(self, c1, c2, c3, c4, size, show_middle=False, show_scores=False):

        c1_size = c1.size()[2:]
        c2_size = c2.size()[2:]
        c3_size = c3.size()[2:]
        c4_size = c4.size()[2:]


        if(show_scores):
            c1_s, c1_b, c1_middle_scores = self.daspp1(c1, show_scores = show_scores)
            c2_s, c2_b, c2_middle_scores = self.daspp2(c2, show_scores = show_scores)
            c3_s, c3_b, c3_middle_scores = self.daspp3(c3, show_scores = show_scores)
            c4_s, c4_b, c4_middle_scores = self.daspp4(c4, show_scores = show_scores)
        else:
            c1_s, c1_b = self.daspp1(c1) 
            c2_s, c2_b = self.daspp2(c2)
            c3_s, c3_b = self.daspp3(c3)
            c4_s, c4_b = self.daspp4(c4)


        if(show_middle):
            middle_features_before_daspp=[c1, c2, c3, c4]
            middle_features_after_daspp=[(c1_s, c1_b), (c2_s, c2_b), (c3_s, c3_b), (c4_s, c4_b)]

        #c1_s_to_b = self.s_to_b_1(c1_s)
        #c2_s_to_b = self.s_to_b_2(c2_s)
        #c3_s_to_b = self.s_to_b_3(c3_s)
        #c4_s_to_b = self.s_to_b_4(c4_s)


        c1_b_to_s = self.b_to_s_1(c1_b)
        c2_b_to_s = self.b_to_s_2(c2_b)
        c3_b_to_s = self.b_to_s_3(c3_b)
        c4_b_to_s = self.b_to_s_4(c4_b)

        #print(torch.max(c1_b_to_s))
        #print(torch.min(c1_b_to_s))

        c4_s_fused = self.conv_s_4(c4_s + c4_s * c4_b_to_s)
        c4_b_fused = self.conv_b_4(c4_b)
        #print(torch.max(c4_b_fused))
        #print(torch.min(c4_b_fused))

        c4_s_fused_up = F.interpolate(c4_s_fused, c3_size, mode='bilinear', align_corners=True)
        c4_b_fused_up = F.interpolate(c4_b_fused, c3_size, mode='bilinear', align_corners=True)

       
        c3_s_fused = self.conv_s_3(c3_s + c3_s * c3_b_to_s + c4_s_fused_up)
        c3_b_fused = self.conv_b_3(c3_b + c4_b_fused_up)

        
        c3_s_fused_up = F.interpolate(c3_s_fused, c2_size, mode='bilinear', align_corners=True)
        c3_b_fused_up = F.interpolate(c3_b_fused, c2_size, mode='bilinear', align_corners=True)
        
        c2_s_fused = self.conv_s_2(c2_s + c2_s * c2_b_to_s + c3_s_fused_up)
        c2_b_fused = self.conv_b_2(c2_b + c3_b_fused_up)
        

        c2_s_fused_up = F.interpolate(c2_s_fused, c1_size, mode='bilinear', align_corners=True)
        c2_b_fused_up = F.interpolate(c2_b_fused, c1_size, mode='bilinear', align_corners=True)
        
        c1_s_fused = self.conv_s_1(c1_s + c1_s * c1_b_to_s + c2_s_fused_up)
        c1_b_fused = self.conv_b_1(c1_b + c2_b_fused_up)


        if(show_middle):
            middle_features_after_fusion=[(c1_s_fused, c1_b_fused), (c2_s_fused, c2_b_fused), (c3_s_fused, c3_b_fused), (c4_s_fused, c4_b_fused)]

        y_s = self.block_s(c1_s_fused)
        y_b = self.block_b(c1_b_fused)
        y_s = F.interpolate(y_s, size, mode='bilinear', align_corners=True)
        y_b = F.interpolate(y_b, size, mode='bilinear', align_corners=True)
        
        if(show_middle):
            return y_s, y_b, middle_features_before_daspp, middle_features_after_daspp, middle_features_after_fusion


        elif(show_scores):
            return y_s, y_b, {'stage1': c1_middle_scores, 'stage2':c2_middle_scores, 'stage3':c3_middle_scores, 'stage4':c4_middle_scores}

        else:
            return y_s, y_b
