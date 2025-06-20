import torch
from torch import nn
from timm import create_model, layers


class ThicknessModel(nn.Module):
    def __init__(self, pretrained=True, backbone='', num_classes=2, dropout=None):
        super(ThicknessModel, self).__init__()
        model = create_model(backbone, pretrained=pretrained)
        features = [*model.children()]
        self.enc = nn.Sequential(*features[:-1])

        # self.classifier = nn.Sequential(
        #     layers.SelectAdaptivePool2d(pool_type='avg'),
        #     layers.LayerNorm2d(model.num_features, eps=1e-06, affine=True),
        #     nn.Flatten(start_dim=1, end_dim=-1),
        #     nn.Dropout(p=dropout, inplace=False),
        #     nn.Linear(in_features=model.num_features, out_features=num_classes, bias=True),
        # )
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(in_features=model.num_features, out_features=num_classes, bias=True)
        )

    def forward(self, x):
        enc = self.enc(x)
        out = self.classifier(enc)
        return out
