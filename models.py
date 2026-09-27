import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2

class ConvBNReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

class PositionAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()
        reduced_channels = max(channels // 8, 1)
        self.query = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.key = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, channels, height, width = x.shape
        query = self.query(x).view(batch, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, height * width)
        attention = self.softmax(torch.bmm(query, key))
        value = self.value(x).view(batch, channels, height * width)
        context = torch.bmm(value, attention.permute(0, 2, 1))
        context = context.view(batch, channels, height, width)
        return x + self.gamma * context

class PFE(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.local_branch = nn.Sequential(ConvBNReLU(in_channels, out_channels), ConvBNReLU(out_channels, out_channels))
        self.global_branch = PositionAttention(in_channels)
        self.fusion = ConvBNReLU(in_channels + out_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        local_feature = self.local_branch(x)
        global_feature = self.global_branch(x)
        return self.fusion(torch.cat([global_feature, local_feature], dim=1))

class ASPP(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0)
        self.branch6 = ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.branch12 = ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.branch18 = ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.fusion = ConvBNReLU(out_channels * 4, out_channels)

    def forward(self, x):
        features = [self.branch1(x), self.branch6(x), self.branch12(x), self.branch18(x)]
        return self.fusion(torch.cat(features, dim=1))

class AMFF(nn.Module):

    def __init__(self, in_channels=256, out_channels=64):
        super().__init__()
        self.fusion = ASPP(in_channels, out_channels)

    def forward(self, features):
        return self.fusion(torch.cat(features, dim=1))

class PFAMNet(nn.Module):

    def __init__(self, num_classes=2, drop_rate=0.2, backbone_weight_path='/workspace/pvt_v2_b2.pth', use_deep_supervision=True):
        super().__init__()
        self.use_deep_supervision = use_deep_supervision
        self.backbone = pvt_v2_b2()
        self._load_backbone_weights(backbone_weight_path)
        self.drop = nn.Dropout2d(drop_rate)
        self.pfe1 = PFE(64, 32)
        self.pfe2 = PFE(128, 64)
        self.pfe3 = PFE(320, 128)
        self.pfe4 = PFE(512, 256)
        self.proj4 = nn.Conv2d(256, 64, kernel_size=1)
        self.proj3 = nn.Conv2d(128, 64, kernel_size=1)
        self.proj2 = nn.Conv2d(64, 64, kernel_size=1)
        self.proj1 = nn.Conv2d(32, 64, kernel_size=1)
        self.refine3 = ConvBNReLU(64, 64)
        self.refine2 = ConvBNReLU(64, 64)
        self.refine1 = ConvBNReLU(64, 64)
        self.amff = AMFF(256, 64)
        self.semantic_head = nn.Conv2d(64, 1, kernel_size=1)
        self.edge_head = nn.Conv2d(64, num_classes, kernel_size=1)
        self.distance_head = nn.Conv2d(64, 1, kernel_size=1)
        if self.use_deep_supervision:
            self.side_head = nn.Conv2d(64, 1, kernel_size=1)

    def _load_backbone_weights(self, path):
        if not path or not os.path.exists(path):
            return
        checkpoint = torch.load(path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        model_state = self.backbone.state_dict()
        pretrained_state = {key: value for key, value in checkpoint.items() if key in model_state and value.shape == model_state[key].shape}
        model_state.update(pretrained_state)
        self.backbone.load_state_dict(model_state)

    def forward(self, x):
        input_size = x.shape[2:]
        x1, x2, x3, x4 = self.backbone(x)
        x1 = self.pfe1(self.drop(x1))
        x2 = self.pfe2(self.drop(x2))
        x3 = self.pfe3(self.drop(x3))
        x4 = self.pfe4(self.drop(x4))
        feat4 = self.proj4(x4)
        feat4_up = F.interpolate(feat4, size=x3.shape[2:], mode='bilinear', align_corners=False)
        side_outputs = []
        if self.use_deep_supervision:
            side_outputs.append(self._side_output(feat4, input_size))
        feat3 = self.refine3(self.proj3(x3) + feat4_up)
        feat3_up = F.interpolate(feat3, size=x2.shape[2:], mode='bilinear', align_corners=False)
        if self.use_deep_supervision:
            side_outputs.append(self._side_output(feat3, input_size))
        feat2 = self.refine2(self.proj2(x2) + feat3_up)
        feat2_up = F.interpolate(feat2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        if self.use_deep_supervision:
            side_outputs.append(self._side_output(feat2, input_size))
        feat1 = self.refine1(self.proj1(x1) + feat2_up)
        if self.use_deep_supervision:
            side_outputs.append(self._side_output(feat1, input_size))
        fuse1 = F.interpolate(feat1, size=input_size, mode='bilinear', align_corners=False)
        fuse2 = F.interpolate(feat2, size=input_size, mode='bilinear', align_corners=False)
        fuse3 = F.interpolate(feat3, size=input_size, mode='bilinear', align_corners=False)
        fuse4 = F.interpolate(feat4, size=input_size, mode='bilinear', align_corners=False)
        fused = self.amff([fuse1, fuse2, fuse3, fuse4])
        mask_out = self.semantic_head(fused)
        edge_out = F.log_softmax(self.edge_head(fused), dim=1)
        dist_out = self.distance_head(fused)
        outputs = [mask_out, edge_out, dist_out]
        if self.use_deep_supervision:
            outputs.extend(side_outputs[::-1])
        return outputs

    def _side_output(self, feature, size):
        feature = F.interpolate(feature, size=size, mode='bilinear', align_corners=False)
        return self.side_head(feature)
if __name__ == '__main__':
    model = PFAMNet()
    inputs = torch.randn(2, 3, 256, 256)
    for output in model(inputs):
        print(output.shape)
