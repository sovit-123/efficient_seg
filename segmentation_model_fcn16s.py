from efficientnet_lite import build_efficientnet_lite
from torchvision.models.feature_extraction import get_graph_node_names

import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNHead32s(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FCNHead32s, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, num_classes, (1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(num_classes),
            # nn.ReLU(True),
            # nn.Dropout(0.1),
            nn.Conv2d(num_classes, num_classes, 1)
        )
    def forward(self, x):
        return self.block(x)
    
class FCNHead16s(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FCNHead16s, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, num_classes, (1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(num_classes),
            # nn.ReLU(True),
            # nn.Dropout(0.1),
            nn.Conv2d(num_classes, num_classes, 1)
        )
    def forward(self, x):
        return self.block(x)

class EffSegModel(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        model_name = 'efficientnet_lite0'
        # EfficientNet model with 1000 out features to load the weights.
        model = build_efficientnet_lite(model_name, 1000)
        if pretrained:
            model.load_state_dict(torch.load(f"weights/{model_name}.pth"))
        self.backbone = nn.Sequential(*list(model.children())[:-4])
        print(self.backbone)
        self.m0 = self.backbone[1][0]
        self.m1 = self.backbone[1][1]
        self.m2 = self.backbone[1][2]
        self.m3 = self.backbone[1][3]
        self.m4 = self.backbone[1][4]
        self.m5 = self.backbone[1][5]
        self.m6 = self.backbone[1][6]

        return_nodes = {
            'layer1': 'layer0',
            'layer2': 'layer1',
            'layer3': 'layer2',
            'layer4': 'layer3',
            'layer4': 'layer4',
            'layer4': 'layer5',
            'layer4': 'layer6',
        }


        self.head = FCNHead32s(320, num_classes)
        
    def forward(self, x):
        size = x.size()[2:]
        x = self.backbone[0](x)
        for layer in self.m0:
            m0_out = layer(x)
        for i, layer in enumerate(self.m1):
            if i == 0:
                m1_out = m0_out
            m1_out = layer(m1_out)
        for i, layer in enumerate(self.m2):
            if i == 0:
                m2_out = m1_out
            m2_out = layer(m2_out)
        for i, layer in enumerate(self.m3):
            if i == 0:
                m3_out = m2_out
            m3_out = layer(m3_out)
        for i, layer in enumerate(self.m4):
            if i == 0:
                m4_out = m3_out
            m4_out = layer(m3_out)
        for i, layer in enumerate(self.m5):
            if i == 0:
                m5_out = m4_out
            m5_out = layer(m4_out)
        for i, layer in enumerate(self.m6):
            if i == 0:
                m6_out = m5_out
            m6_out = layer(m6_out)

        x = m5_out + m6_out
        
        x = self.head(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return x
        
    
if __name__ == '__main__':
    model = EffSegModel(num_classes=2)
    # print(model)
    # Total parameters and trainable parameters.
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"{total_params:,} total parameters.")
    # total_trainable_params = sum(
        # p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"{total_trainable_params:,} training parameters.")
    # model.eval()
    # tensor = torch.rand((1, 3, 512, 512))
    # output = model(tensor)
    # print(f"Output shape: {output.shape}")
    # print('*'*50)
    # print('Individual Sequential layers)
    # for i, layer in enumerate(model.backbone[1]):
    #     print(f"LAYER {i}, {layer}")