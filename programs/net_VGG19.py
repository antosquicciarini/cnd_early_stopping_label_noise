import torch
import torch.nn as nn
import torch.nn.functional as F
from dropout import ExampleTiedDropout
import torchvision
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG19_OLD(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):

        super(VGG19_OLD, self).__init__()
        
        # Define each block of the VGG19 architecture as separate Sequential modules
        self.block1 = nn.Sequential(
            conv_bn(input_channels, 64),
            conv_bn(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block2 = nn.Sequential(
            conv_bn(64, 128),
            conv_bn(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block3 = nn.Sequential(
            conv_bn(128, 256),
            conv_bn(256, 256),
            conv_bn(256, 256),
            conv_bn(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block4 = nn.Sequential(
            conv_bn(256, 512),
            conv_bn(512, 512),
            conv_bn(512, 512),
            conv_bn(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block5 = nn.Sequential(
            conv_bn(512, 512),
            conv_bn(512, 512),
            conv_bn(512, 512),
            conv_bn(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully connected layers
        self.FCN_1 = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout()
        )

        self.FCN_2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )

        self.classifier = nn.Linear(4096, num_classes)

    def forward(self, x, idx, return_intermediates=False, neuron_indexes=[0], layer_indexes=[], apply_mask=False):
        # Pass through each block sequentially

        def rescale_output(x, neuron_indexes, intermediates):
            out_int = F.avg_pool2d(x, x.shape[2])
            out_int = out_int.view(out_int.size(0), -1)
            # In order to reduce computation time, select only 20% of
            intermediates.append(out_int[:, neuron_indexes])  #intermediates.append(out_int[:, neuron_indexes])
            return intermediates

        intermediates = []
        # layer_indexes = [0]
        # return_intermediates=True
        # neuron_indexes = slice(None)

        x = self.block1(x)
        if return_intermediates:
            intermediates = rescale_output(x, neuron_indexes, intermediates)

        x = self.block2(x)
        if return_intermediates:
            intermediates = rescale_output(x, neuron_indexes, intermediates)

        x = self.block3(x)
        if return_intermediates:
            intermediates = rescale_output(x, neuron_indexes, intermediates)

        x = self.block4(x)
        if return_intermediates:
            intermediates = rescale_output(x, neuron_indexes, intermediates)

        x = self.block5(x)
        if return_intermediates:
            intermediates = rescale_output(x, neuron_indexes, intermediates)

        
        # Flatten and pass through the classifier
        x = x.squeeze()
        x = self.FCN_1(x)
        if return_intermediates:
            intermediates.append(x[:, neuron_indexes])

        x = self.FCN_2(x)
        if return_intermediates: # and (isinstance(layer_indexes, slice) or 6 in layer_indexes)
            intermediates.append(x[:, neuron_indexes])

        x = self.classifier(x)

        if return_intermediates:
            return x, torch.cat(intermediates, dim=1)
          
        return x

# Helper function to create a convolutional layer with batch norm and ReLU
def conv_bn(in_channels, out_channels, kernel_size=3, stride=1, padding=1, apply_Relu=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels)
    ]
    if apply_Relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)






class VGG19(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):

        super(VGG19, self).__init__()
        
        # Define each block of the VGG19 architecture as separate Sequential modules
        self.block1 = nn.Sequential(
            conv_bn(input_channels, 64),
            conv_bn(64, 64, apply_Relu=False)
        )
        
        self.block2 = nn.Sequential(
            conv_bn(64, 128),
            conv_bn(128, 128, apply_Relu=False)
        )

        self.block3 = nn.Sequential(
            conv_bn(128, 256),
            conv_bn(256, 256),
            conv_bn(256, 256),
            conv_bn(256, 256, apply_Relu=False)
        )
        
        self.block4 = nn.Sequential(
            conv_bn(256, 512),
            conv_bn(512, 512),
            conv_bn(512, 512),
            conv_bn(512, 512, apply_Relu=False)
        )
        
        self.block5 = nn.Sequential(
            conv_bn(512, 512),
            conv_bn(512, 512),
            conv_bn(512, 512),
            conv_bn(512, 512, apply_Relu=False)
        )
        
        self.FCN_1 = nn.Linear(512, 4096)
        self.FCN_2 = nn.Linear(4096, 4096)
        self.classifier = nn.Linear(4096, num_classes)

    def forward(self, x, idx, return_intermediates=False, neuron_indexes=[0], layer_indexes=[], apply_mask=False):
        # Pass through each block sequentially

        def rescale_output(x, neuron_indexes, intermediates):
            out_int = F.max_pool2d(x, x.shape[2])  # Replace avg_pool2d with max_pool2d
            out_int = out_int.view(out_int.size(0), -1)
            intermediates.append(out_int[:, neuron_indexes])  # Select only specified neurons
            return intermediates

        intermediates = []
        # layer_indexes = [0]
        # return_intermediates=True
        # neuron_indexes = slice(None)

        x = self.block1(x)
        if return_intermediates:
            intermediates = rescale_output(x, neuron_indexes, intermediates)
        x = nn.ReLU(inplace=True)(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = self.block2(x)
        if return_intermediates:
            intermediates = rescale_output(x, neuron_indexes, intermediates)
        x = nn.ReLU(inplace=True)(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = self.block3(x)
        if return_intermediates:
            intermediates = rescale_output(x, neuron_indexes, intermediates)
        x = nn.ReLU(inplace=True)(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = self.block4(x)
        if return_intermediates:
            intermediates = rescale_output(x, neuron_indexes, intermediates)
        x = nn.ReLU(inplace=True)(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = self.block5(x)
        if return_intermediates:
            intermediates = rescale_output(x, neuron_indexes, intermediates)
        x = nn.ReLU(inplace=True)(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        # Flatten and pass through the classifier
        x = x.squeeze()
        x = self.FCN_1(x)
        if return_intermediates:
            intermediates.append(x[:, neuron_indexes])
        x = nn.ReLU(True)(x)
        x = nn.Dropout()(x)

        x = self.FCN_2(x)
        if return_intermediates: # and (isinstance(layer_indexes, slice) or 6 in layer_indexes)
            intermediates.append(x[:, neuron_indexes])
        x = nn.ReLU(True)(x)
        x = nn.Dropout()(x)

        x = self.classifier(x)

        if return_intermediates:
            return x, torch.cat(intermediates, dim=1)
          
        return x

# Helper function to create a convolutional layer with batch norm and ReLU
def conv_bn(in_channels, out_channels, kernel_size=3, stride=1, padding=1, apply_Relu=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels)
    ]
    if apply_Relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)