from efficientnet_lite import build_efficientnet_lite

import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, num_classes, 1)
        )

    def forward(self, x):
        return self.block(x)

class EffSegModel(nn.Module):
    def __init__(self, num_classes=1, pretrained=True, aux=False):
        super().__init__()
        model_name = 'efficientnet_lite1'
        # EfficientNet model with 1000 out features to load the weights.
        model = build_efficientnet_lite(model_name, 1000)
        if pretrained:
            model.load_state_dict(torch.load(f"models/efficientnet_weights/{model_name}.pth"))
        self.backbone = nn.Sequential(*list(model.children())[:-4])        
        
        self.aux_out = FCNHead(192, num_classes) if aux else None
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
                    aux_cls = layer(x)
                x = layer(x)
        
        if self.aux_out is not None:
            aux = self.aux_out(aux_cls)
            aux_out = F.interpolate(aux, size, mode='bilinear', align_corners=True)
        else:
            aux_out = None
        results['aux'] = aux_out
        x = self.head(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        results['out'] = x
        return results
        
    
if __name__ == '__main__':
    aux = True
    model = EffSegModel(num_classes=2, aux=aux)
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    model.eval()
    tensor = torch.rand((1, 3, 512, 512))
    output = model(tensor)
    print(f"Output shape: {output['out'].shape}, {output['aux'].shape if aux else None}")
    # print('*'*50)
    # print('Individual Sequential layers)
    # for i, layer in enumerate(model.backbone[1]):
    #     print(f"LAYER {i}, {layer}")