"""Small convolutional classifier with global average pooling."""

import torch
from torch import nn


class BirdCNN(nn.Module):
    def __init__(self, num_classes, channels=(32, 64, 128, 256), convolutions_per_stage=2):
        super().__init__()
        if num_classes < 2 or not channels or any(c < 1 for c in channels) or convolutions_per_stage < 1:
            raise ValueError('Invalid CNN architecture')
        stages = []
        inputs = 3
        for outputs in channels:
            layers = []
            for layer in range(convolutions_per_stage):
                layers.extend([nn.Conv2d(inputs, outputs, 3, stride=2 if layer == 0 else 1,
                                         padding=1, bias=False), nn.BatchNorm2d(outputs), nn.ReLU(inplace=True)])
                inputs = outputs
            stages.append(nn.Sequential(*layers))
        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels[-1], num_classes)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.classifier(torch.flatten(self.pool(self.features(x)), 1))
