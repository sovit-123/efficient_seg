import torch.nn as nn

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