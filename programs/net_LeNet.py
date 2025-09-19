import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, dropout_rate=0.0):
        super(LeNet, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)  # Dropout for fully connected layers
        self.spatial_dropout = nn.Dropout2d(dropout_rate)  # Spatial dropout for convolutional layers

        # Define the layers of LeNet-5
        self.conv1 = nn.Conv2d(input_channels, 20, kernel_size=5, stride=1, padding=0)  # 28x28 -> 24x24
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 24x24 -> 12x12
        
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0)  # 12x12 -> 8x8
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8 -> 4x4

        self.fc1 = nn.Linear(1250, 500)  # Flattened size: 50*4*4 = 800
        self.fc2 = nn.Linear(500, num_classes)  # Output size: num_classes (e.g., 10)

    def set_mask(self, mask):
        self.masks = mask
    def disable_mask(self):
        self.masks = None

    def record_intermediate(self, x, intermediates, por_neuron):
        pooled = self.adaptive_pool(x).view(x.size(0), -1)
        neuron_reduced_index = int(pooled.shape[1] * por_neuron)
        intermediates.append(pooled[:, :neuron_reduced_index])
        return intermediates
    
    def apply_mask(self, x, mask):
        """
        Applies a boolean mask to the activation tensor `x`.

        Parameters:
        - x (torch.Tensor): The activation tensor. Can be 2D or 4D.
        - mask (torch.Tensor): The boolean mask of shape (C,), where C is the channel dimension of `x`.

        Returns:
        - torch.Tensor: The masked activation tensor.
        """
        if x.dim() == 2:
            return x * mask
        elif x.dim() == 4:
            return x * mask.view(1, -1, 1, 1)  # Reshape mask for broadcasting
        else:
            raise ValueError(f"Unsupported tensor dimensions: {x.shape}")
        
    def forward(self, x, idx, return_intermediates=False, neuron_indexes=slice(None), layer_indexes=[], apply_mask=False, CND_reg_only_last_layer=False, por_neuron=1.0):
        intermediates = []

        # Layer 1: Conv1 + Dropout + MaxPool1
        out = self.conv1(x)
        if apply_mask and 0 in self.masks:
            out = self.apply_mask(out, self.masks[0])
        if return_intermediates and not CND_reg_only_last_layer:
            intermediates = self.record_intermediate(out, intermediates, por_neuron)
        out = F.relu(out)
        out = self.spatial_dropout(out)  # Apply spatial dropout
        out = self.pool1(out)  # Max Pooling

        # Layer 2: Conv2 + Dropout + MaxPool2
        out = self.conv2(out)
        if apply_mask and 1 in self.masks:
            out = self.apply_mask(out, self.masks[1])
        if return_intermediates and not CND_reg_only_last_layer:
            intermediates = self.record_intermediate(out, intermediates, por_neuron)
        out = F.relu(out)
        out = self.spatial_dropout(out)  # Apply spatial dropout
        out = self.pool2(out)  # Max Pooling

        # Flatten
        out = out.view(out.size(0), -1)  # Flatten to feed into fully connected layers

        # Fully connected layer 1 + Dropout
        out = self.dropout(out)
        out = self.fc1(out)
        if apply_mask and 2 in self.masks:
            out = self.apply_mask(out, self.masks[2])
        if return_intermediates and not CND_reg_only_last_layer:
            intermediates = self.record_intermediate(out, intermediates, por_neuron)
        out = F.relu(out)

        # Fully connected layer 2
        out = self.dropout(out)
        out = self.fc2(out)

        # Return results
        if return_intermediates:
            return out, torch.cat(intermediates, dim=1)
        else:
            return out, None
    