import torch
import torch.nn as nn
import torch.nn.functional as F
from dropout import ExampleTiedDropout
import torchvision
import torchvision.models as models




class Mul(nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight
    def forward(self, x): return x * self.weight

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x): return x + self.module(x)

def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return nn.Sequential(
            nn.Conv2d(channels_in, channels_out,
                         kernel_size=kernel_size, stride=stride, padding=padding,
                         groups=groups, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU()
    )


class ResNet9_dropout(nn.Module):
    def __init__(self, device, p_fixed = 0.2, p_mem = 0.1, num_batches = 100, drop_mode = "train", input_channels = 3, fac = 1, num_classes = 10):
        super(ResNet9_dropout, self).__init__()
        self.p_fixed = p_fixed
        self.p_mem = p_mem
        self.num_batches = num_batches
        self.drop_mode = drop_mode

        dims = [input_channels, 64, 128, 128, 128, 256, 256, 256,128]
        dims = [int(d*fac) for d in dims]
        dims[0] = input_channels
        self.dims = dims

        self.dropout0 = self.get_dropout(device, p_fixed, p_mem, num_batches, drop_mode)
        self.dropout1 = self.get_dropout(device, p_fixed, p_mem, num_batches, drop_mode)
        self.dropout2 = self.get_dropout(device, p_fixed, p_mem, num_batches, drop_mode)
        self.dropout3 = self.get_dropout(device, p_fixed, p_mem, num_batches, drop_mode)
        self.dropout4 = self.get_dropout(device, p_fixed, p_mem, num_batches, drop_mode)
        self.dropout5 = self.get_dropout(device, p_fixed, p_mem, num_batches, drop_mode)


        self.conv1 = conv_bn(dims[0], dims[1], kernel_size=3, stride=1, padding=1)
        self.conv2 = conv_bn(dims[1], dims[2], kernel_size=5, stride=2, padding=2)
        self.res1 = Residual(nn.Sequential(conv_bn(dims[2], dims[3]), conv_bn(dims[3], dims[4])))
        self.conv3 = conv_bn(dims[4], dims[5], kernel_size=3, stride=1, padding=1)
        self.res2 = Residual(nn.Sequential(conv_bn(dims[5], dims[6]), conv_bn(dims[6], dims[7])))
        self.conv4 = conv_bn(dims[7], dims[8], kernel_size=3, stride=1, padding=0)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.maxpool = nn.MaxPool2d(2)
        self.flatten = Flatten()
        self.linear = nn.Linear(dims[8], num_classes, bias=False)
        self.mul  = Mul(0.2)

    # def train(self, mode=True):
    #     """Override the train method to automatically set drop_mode to 'train'."""
    #     super(ResNet9_dropout, self).train(mode)  # Call the parent's train method
    #     if mode:
    #         self.set_drop_mode("train")
    #     else:
    #         self.set_drop_mode("test")

    # def eval(self):
    #     """Override the eval method to automatically set drop_mode to 'test'."""
    #     super(ResNet9_dropout, self).eval()  # Call the parent's eval method
    #     self.set_drop_mode("test")

    def set_drop_mode(self, mode):
        self.drop_mode = mode
        self.dropout0.drop_mode = mode
        self.dropout1.drop_mode = mode
        self.dropout2.drop_mode = mode
        self.dropout3.drop_mode = mode
        self.dropout4.drop_mode = mode
        self.dropout5.drop_mode = mode

    def get_dropout(self, device, p_fixed, p_mem, num_batches, drop_mode):
        return ExampleTiedDropout(device, p_fixed=p_fixed, p_mem=p_mem,num_batches=num_batches, drop_mode = drop_mode)
        #return nn.Dropout(p=p_fixed)

    def forward(self, x, idx, return_intermediates=False, neuron_indexes=[0], layer_indexes=[], apply_mask=False):

        intermediates = []

        x = self.conv1(x)
        x = self.dropout0(x, idx)

        if return_intermediates and 0 in layer_indexes:
            out_int = F.avg_pool2d(x, x.shape[2])
            out_int = out_int.view(out_int.size(0), -1)
            intermediates.append(out_int[:, neuron_indexes])

        x = self.conv2(x)
        x = self.dropout1(x, idx)

        if return_intermediates and 1 in layer_indexes:
            out_int = F.avg_pool2d(x, x.shape[2])
            out_int = out_int.view(out_int.size(0), -1)
            intermediates.append(out_int[:, neuron_indexes])

        x = self.res1(x)
        x = self.dropout2(x, idx)
        if return_intermediates and 2 in layer_indexes:
            out_int = F.avg_pool2d(x, x.shape[2])
            out_int = out_int.view(out_int.size(0), -1)
            intermediates.append(out_int[:, neuron_indexes])

        x = self.conv3(x)
        x = self.dropout3(x, idx)
        if return_intermediates and 3 in layer_indexes:
            out_int = F.avg_pool2d(x, x.shape[2])
            out_int = out_int.view(out_int.size(0), -1)
            intermediates.append(out_int[:, neuron_indexes])

        x = self.maxpool(x)

        x = self.res2(x)
        x = self.dropout4(x, idx)
        if return_intermediates and 4 in layer_indexes:
            out_int = F.avg_pool2d(x, x.shape[2])
            out_int = out_int.view(out_int.size(0), -1)
            intermediates.append(out_int[:, neuron_indexes])

        x = self.conv4(x)
        x = self.dropout5(x, idx)
        if return_intermediates and 5 in layer_indexes:
            out_int = F.avg_pool2d(x, x.shape[2])
            out_int = out_int.view(out_int.size(0), -1)
            intermediates.append(out_int[:, neuron_indexes])

        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.mul(x)
        
        if return_intermediates:
            return x, torch.cat(intermediates, dim=1)

        return x    




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, last_block_layer=False, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        self.last_block_layer = last_block_layer
        self.spatial_dropout = nn.Dropout2d(dropout_rate)  # Spatial dropout for convolutional layers

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.spatial_dropout(out)  # Apply spatial dropout after first convolution
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if self.last_block_layer:
            return out
        else:
            return F.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, last_block_layer=False, dropout_rate=0.0):
        super(Bottleneck, self).__init__()
        self.last_block_layer = last_block_layer
        self.dropout = nn.Dropout2d(dropout_rate)  # Spatial Dropout

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)  # Apply spatial dropout after first convolution
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)  # Apply spatial dropout after second convolution
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        if self.last_block_layer:
            return out
        else:
            return F.relu(out)

    
import torch
import torch.nn as nn
import torch.nn.functional as F
from dropout import ExampleTiedDropout  # Assuming this is defined elsewhere
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, implement_exp_tied_dropout=False, num_classes=10,
                 p_fixed=0.2, p_mem=0.1, num_batches=100, drop_mode="train",
                 input_channels=3, fac=1, dropout_rate=0.0, network=None, implement_pre_act=False):
        super(ResNet, self).__init__()

        self.implement_exp_tied_dropout = implement_exp_tied_dropout
        self.num_classes = num_classes
        self.in_planes = 64

        # Predefine a max-pooling layer (used for 224x224 inputs)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if network == 'TinyImagenet':
            self.layer0 = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            self.layer0 = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if self.implement_exp_tied_dropout:
            self.exp_tied_dropout = ExampleTiedDropout(p_fixed=p_fixed, p_mem=p_mem,
                                                       num_batches=num_batches, drop_mode=drop_mode)

        self.masks = {}
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.spatial_dropout = nn.Dropout2d(dropout_rate)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, s in enumerate(strides):
            # Only record pre-activation on the last block if needed
            last_block_layer = (i == len(strides) - 1)
            layers.append(block(self.in_planes, planes, s, last_block_layer=last_block_layer))
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

        # Initial convolution block
        out = self.layer0(x)
        # If input size is 224, apply maxpool as defined in __init__
        if x.size(2) == 224:
            out = self.maxpool(out)

        # Layer 1
        pre_act = self.layer1(out)
        if apply_mask and 0 in self.masks:
            pre_act = self.apply_mask(pre_act, self.masks[0])
        if return_intermediates and not CND_reg_only_last_layer:
            intermediates = self.record_intermediate(pre_act, intermediates, por_neuron)
        out = F.relu(pre_act, inplace=True)
        out = self.spatial_dropout(out)

        # Layer 2
        pre_act = self.layer2(out)
        if apply_mask and 1 in self.masks:
            pre_act = self.apply_mask(pre_act, self.masks[1])
        if return_intermediates and not CND_reg_only_last_layer:
            intermediates = self.record_intermediate(pre_act, intermediates, por_neuron)
        out = F.relu(pre_act, inplace=True)
        out = self.spatial_dropout(out)
        if self.implement_exp_tied_dropout and idx is not None:
            out = self.exp_tied_dropout(out, idx)

        # Layer 3
        pre_act = self.layer3(out)
        if apply_mask and 2 in self.masks:
            pre_act = self.apply_mask(pre_act, self.masks[2])
        if return_intermediates and not CND_reg_only_last_layer:
            intermediates = self.record_intermediate(pre_act, intermediates, por_neuron)
        out = F.relu(pre_act, inplace=True)
        out = self.spatial_dropout(out)
        if self.implement_exp_tied_dropout and idx is not None:
            out = self.exp_tied_dropout(out, idx)

        # Layer 4
        pre_act = self.layer4(out)

        if apply_mask and 3 in self.masks:
            pre_act = self.apply_mask(pre_act, self.masks[3])
        if return_intermediates:
            intermediates = self.record_intermediate(pre_act, intermediates, por_neuron)

        out = F.relu(pre_act, inplace=True)

        # Global pooling and classification
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)

        if return_intermediates:
            return out, torch.cat(intermediates, dim=1)
        return out, None

    def set_mask(self, mask):
        self.masks = mask

    def disable_mask(self):
        self.masks = {}

# Optionally, compile the model using TorchScript for further efficiency improvements:
# scripted_model = torch.jit.script(ResNet(...))




# class ResNet_Fast(nn.Module):
#     def __init__(self, args):
#         super(ResNet_Fast, self).__init__()

#         # Load ResNet-34
#         if args.network == "ResNet34_Fast":
#             self.resnet = models.resnet34(pretrained=False)

#         # Load ResNet-50
#         if args.network == "ResNet50_Fast":
#             self.resnet = models.resnet50(pretrained=False)

#         # Modify the final fully connected layer (fc) 
#         num_ftrs = self.resnet.fc.in_features  # Get the input features of the last layer
#         self.resnet.fc = nn.Identity()         # Remove the original fc layer by setting it to identity
#         self.fc = nn.Linear(num_ftrs, args.num_classes)  # Create a new fully connected layer


#     def forward(self, x, idx=None, return_intermediates=False, CND_reg_only_last_layer=True, apply_mask=False, layer_indexes=None,  por_neuron=1.0):
#         """
#         Forward pass. If `return_intermediates=True`, also return the pre-activation of the penultimate layer.
#         """
    
#         preact = self.resnet(x)         # Obtain features from the modified ResNet (without fc)
#         logits = self.fc(preact)          # Compute final output using the new fc layer
        
#         if return_intermediates:
#             neuron_reduced_index = int(preact.shape[1] * por_neuron)
#             return logits, preact[:,:neuron_reduced_index]       # Return both logits and pre-activation features
#         return logits, None


#     def set_mask(self, mask): #For compatibility
#         self.masks = mask

#     def disable_mask(self): #For compatibility
#         self.masks = {}

class ResNet_Fast(nn.Module):
    def __init__(self, args):
        super(ResNet_Fast, self).__init__()
        # Load ResNet-34 or ResNet-50 based on args
        if args.network == "ResNet34_Fast":
            self.resnet = models.resnet34(pretrained=False)
        elif args.network == "ResNet50_Fast":
            self.resnet = models.resnet50(pretrained=False)
        else:
            raise ValueError("Unsupported network type")
        
        # Remove the original fc layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        # New final fully-connected layer for classification
        self.fc = nn.Linear(num_ftrs, args.num_classes)
        # Define an adaptive pooling layer for recording pre-activations
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))
    
    def record_intermediate(self, x, por_neuron):
        # Pool the feature maps and flatten to [batch, channels]
        pooled = self.adaptive_pool(x).view(x.size(0), -1)
        # Select a fraction (por_neuron) of the neurons
        neuron_reduced_index = int(pooled.shape[1] * por_neuron)
        return pooled[:, :neuron_reduced_index]
    
    def forward(self, x, indx, return_intermediates=False, CND_reg_only_last_layer=False, apply_mask=False, layer_indexes=None, por_neuron=1.0):
        """
        Args:
            x: input tensor
            return_intermediates: if True, returns intermediate pre-activations.
            layer_indexes: optional list of layer indexes to record (0:layer1, 1:layer2, etc.)
            por_neuron: fraction of neurons to keep from the pooled activations.
        """
        intermediates = []
        # Follow the standard ResNet architecture steps:
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        # Layer 1
        out1 = self.resnet.layer1(x)
        if return_intermediates and not CND_reg_only_last_layer:
            intermediates.append(self.record_intermediate(out1, por_neuron))
        
        # Layer 2
        out2 = self.resnet.layer2(out1)
        if return_intermediates and not CND_reg_only_last_layer:
            intermediates.append(self.record_intermediate(out2, por_neuron))
        
        # Layer 3
        out3 = self.resnet.layer3(out2)
        if return_intermediates and not CND_reg_only_last_layer:
            intermediates.append(self.record_intermediate(out3, por_neuron))
        
        # Layer 4
        out4 = self.resnet.layer4(out3)
        if return_intermediates:
            intermediates.append(self.record_intermediate(out4, por_neuron))
        
        # Global average pooling and classification head
        out = self.resnet.avgpool(out4)
        out = torch.flatten(out, 1)
        logits = self.fc(out)
        
        if return_intermediates:
            # Concatenate the intermediate activations along the feature dimension
            return logits, torch.cat(intermediates, dim=1)
        
        return logits, None


class ResNet50p(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50p, self).__init__()
        # Load the pre-defined ResNet50 model
        self.resnet50 = models.resnet50(pretrained=False, num_classes=num_classes)
    
    def forward(self, x, sample_indexes):
        # x: input features (images)
        # sample_indexes: additional input (sample indexes), which we ignore
        output = self.resnet50(x)  # Forward pass with the original input (ignore sample_indexes)
        return output
