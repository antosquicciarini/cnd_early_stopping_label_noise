
'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from dropout import ExampleTiedDropout
import torchvision
import torchvision.models as models


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, implement_exp_tied_dropout=False, num_classes=10,
                 p_fixed=0.2, p_mem=0.1, num_batches=100, drop_mode="train",
                 input_channels=3, fac=1, dropout_rate=0.0, network=None, implement_pre_act=False):
        super(PreActResNet, self).__init__()


        self.implement_exp_tied_dropout = implement_exp_tied_dropout
        self.num_classes = num_classes
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def apply_mask(self, x, mask):
        # Normalize activations based on percentage of active neurons
        p_on = torch.sum(mask) / mask.numel()
        p_on = p_on if p_on > 0 else 1
        x = x / p_on
        if x.dim() == 2:
            return x * mask
        elif x.dim() == 4:
            return x * mask.view(1, -1, 1, 1)
        else:
            raise ValueError(f"Unsupported tensor dimensions: {x.shape}")

    def record_intermediate(self, x, intermediates, por_neuron):
        pooled = self.adaptive_pool(x).view(x.size(0), -1)
        neuron_reduced_index = int(pooled.shape[1] * por_neuron)
        intermediates.append(pooled[:, :neuron_reduced_index])
        return intermediates

    def forward(self, x, idx=None, return_intermediates=False, layer_indexes=[], apply_mask=False,
                CND_reg_only_last_layer=False, por_neuron=1.0):        
        
        intermediates = []

        out = self.conv1(x)

        out = self.layer1(out)
        if apply_mask and 0 in self.masks:
            out = self.apply_mask(out, self.masks[0])
        if return_intermediates and not CND_reg_only_last_layer:
            intermediates = self.record_intermediate(out, intermediates, por_neuron)
        out = self.spatial_dropout(out)

        out = self.layer2(out)
        if apply_mask and 1 in self.masks:
            out = self.apply_mask(out, self.masks[1])
        if return_intermediates and not CND_reg_only_last_layer:
            intermediates = self.record_intermediate(out, intermediates, por_neuron)
        out = self.spatial_dropout(out)
        if self.implement_exp_tied_dropout and idx is not None:
            out = self.exp_tied_dropout(out, idx)

        out = self.layer3(out)
        if apply_mask and 2 in self.masks:
            out = self.apply_mask(out, self.masks[2])
        if return_intermediates and not CND_reg_only_last_layer:
            intermediates = self.record_intermediate(out, intermediates, por_neuron)
        out = self.spatial_dropout(out)
        if self.implement_exp_tied_dropout and idx is not None:
            out = self.exp_tied_dropout(out, idx)

        out = self.layer4(out)
        if apply_mask and 3 in self.masks:
            pre_act = self.apply_mask(pre_act, self.masks[3])
        if return_intermediates:
            intermediates = self.record_intermediate(pre_act, intermediates, por_neuron)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        if return_intermediates:
            return out, torch.cat(intermediates, dim=1)
        return out, None