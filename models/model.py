#coding:utf-8
from addict import Dict
from torch import nn
import torch.nn.functional as F

# from models.backbone import build_backbone
# from models.neck import build_neck
# from models.head import build_head
from .backbone import build_backbone
from .neck import build_neck
from .head import build_head

class Model(nn.Module):
    def __init__(self, model_config: dict):
        """
        PANnet
        :param model_config: 模型配置
        """
        super().__init__()
        model_config = Dict(model_config)
        backbone_type = model_config.backbone.pop('type')
        neck_type = model_config.neck.pop('type')
        head_type = model_config.head.pop('type')
        self.backbone = build_backbone(backbone_type, **model_config.backbone)
        self.neck = build_neck(neck_type, in_channels=self.backbone.out_channels, **model_config.neck)
        self.head = build_head(head_type, in_channels=self.neck.out_channels, **model_config.head)
        self.name = f'{backbone_type}_{neck_type}_{head_type}'

    def forward(self, x):
        _, _, H, W = x.size()
        # print('==H, W:', H, W)
        H, W = int(H), int(W)
        backbone_out = self.backbone(x)
        # for i in backbone_out:
        #     print('backbone=====', i.shape)
        neck_out = self.neck(backbone_out)
        # for i in neck_out:
        #     print('neck=====', i.shape)
        y = self.head(neck_out)
        # y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)
        y = F.interpolate(y, size=(H, W))
        return y


if __name__ == '__main__':
    import torch

    # device = torch.device('cpu')
    x = torch.rand((8, 3, 640, 640)).cuda()

    model_config = {
        'backbone': {'type': 'resnet18', 'pretrained': False, "in_channels": 3},
        'neck': {'type': 'FPN', 'inner_channels': 256},  # 分割头，FPN or FPEM_FFM
        'head': {'type': 'DBHead', 'out_channels': 2, 'k': 50},
    }
    model = Model(model_config=model_config).cuda()
    import time

    print(model)
    tic = time.time()
    y = model(x)

    print('y.shape:', y.shape)
    print(model.name)

    # torch.save(model.state_dict(), 'PAN.pth')
