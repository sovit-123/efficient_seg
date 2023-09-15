from models.efficientnet_lite import build_efficientnet_lite
from models.fcn_decoder import FCNHead

import torch
import torch.nn as nn
import torch.nn.functional as F

class EffSegModel(nn.Module):
    def __init__(self, num_classes=1, pretrained=True, aux=False):
        super().__init__()
        self.aux = aux
        model_name = 'efficientnet_lite0'
        # EfficientNet model with 1000 out features to load the weights.
        model = build_efficientnet_lite(model_name, 1000)
        if pretrained:
            model.load_state_dict(torch.load(f"models/efficientnet_weights/{model_name}.pth"))
        self.backbone = nn.Sequential(*list(model.children())[:-4])        
        
        self.aux_out = FCNHead(192, num_classes) if self.aux else None
        self.module5_conv = nn.Conv2d(
            192, 320, kernel_size=1, stride=1, padding=0
        )
        self.head = FCNHead(320, num_classes)
        
    def forward(self, x):
        results = {}
        size = x.size()[2:]
        x = self.backbone[0](x)
        counter = 0
        for module_layer in self.backbone[1]:
            counter += 1
            for layer in module_layer:
                if counter == 6:
                    if self.aux:
                        aux_cls = layer(x)
                    module5 = layer(x)
                x = layer(x)
        
        if self.aux_out is not None:
            aux = self.aux_out(aux_cls)
            aux_output = F.interpolate(
                aux, size, mode='bilinear', align_corners=False
            )
        else:
            aux_output = None
        results['aux'] = aux_output

        module5 = self.module5_conv(module5)
        x += module5
        x = self.head(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=False)
        results['out'] = x
        return results