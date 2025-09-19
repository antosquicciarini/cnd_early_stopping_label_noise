import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.models as models

from dropout import ExampleTiedDropout



class FullyConnectedNN(nn.Module):
    def __init__(self, device, input_channels=1, input_size=28, L=3, N=128, batch_normalization_flag=False, dropout_rate=0.0):
        super(FullyConnectedNN, self).__init__()
        self.input_view = input_channels * input_size * input_size
        self.L = L  # Number of layers
        self.N = N  # Neurons per hidden layer
        self.batch_normalization_flag = batch_normalization_flag
        self.device = device
        self.dropout_rate = dropout_rate

        # Create a list to hold the fully connected layers and batch normalization layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_normalization_flag else None
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer

        if batch_normalization_flag:
            self.batch_norms.append(nn.BatchNorm1d(N))
            
        # First layer from input to first hidden layer
        self.layers.append(nn.Linear(self.input_view, N))

        if batch_normalization_flag:
            self.batch_norms.append(nn.BatchNorm1d(N))

        # Hidden layers
        for _ in range(L-1):
            self.layers.append(nn.Linear(N, N))
            if batch_normalization_flag:
                self.batch_norms.append(nn.BatchNorm1d(N))

        # Final output layer (assuming 10 output classes as in your original code)
        self.output_layer = nn.Linear(N, 10)

    def apply_mask(self, x, mask):
        return x * mask

    def forward(self, x, idx, return_intermediates=False, neuron_indexes=[0], layer_indexes=[], apply_mask=False):
        x = x.view(-1, self.input_view)  # Flatten the input
        intermediates = []

        # Pass through each layer
        for i, layer in enumerate(self.layers):
            if apply_mask:
                x = torch.relu(self.apply_mask(layer(x), self.mask[:self.N]))
            else:
                x = torch.relu(layer(x))

            # Apply batch normalization if required
            if self.batch_normalization_flag:
                x = self.batch_norms[i](x)

            # Apply dropout
            x = self.dropout(x)

            # Check if layer_indexes is a slice and treat it as selecting all layers if so
            if return_intermediates:
                if isinstance(layer_indexes, slice) or i in layer_indexes:
                    intermediates.append(x[:, neuron_indexes])

        # Final output layer (no activation)
        x = self.output_layer(x)

        if return_intermediates:
            return x, torch.cat(intermediates, dim=1)
        return x

import torch.nn.functional as F

class NewsMLP(nn.Module):
    """
    3-layer MLP for the NEWS dataset with fixed GloVe embeddings,
    adaptive average pooling, batch normalization, Softsign activation,
    and return_intermediates functionality.

    TODO - Implement mask capabilities
    """
    def __init__(
        self,
        device,
        embedding_weights,
        embedding_dim=300,
        num_classes=7,
        dropout_rate=0.0
    ):
        super(NewsMLP, self).__init__()
        self.device = device
        # Embedding layer (frozen)
        embedding_weights = torch.tensor(embedding_weights, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(
            embedding_weights, freeze=True
        )
        # Adaptive pooling to reduce sequence length to 20 (not 16 because non-divisible input sizes are not implemented on MPS device yet.)
        self.pool = nn.AdaptiveAvgPool1d(20)
        #self.avgpool=nn.AdaptiveAvgPool1d(20*embedding_dim) #TODO delete this

        # First dense: 16*embedding_dim -> 4*embedding_dim
        self.fc1 = nn.Linear(20 * embedding_dim, 4 * embedding_dim)
        self.bn1 = nn.BatchNorm1d(4 * embedding_dim)
        # Second dense: 4*embedding_dim -> embedding_dim
        self.fc2 = nn.Linear(4 * embedding_dim, embedding_dim)
        self.bn2 = nn.BatchNorm1d(embedding_dim)

        # Final classification layer: embedding_dim -> num_classes
        self.fc3 = nn.Linear(embedding_dim, num_classes)
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)


    def apply_mask(self, x, mask):
        p_on = torch.sum(mask) / mask.numel()  # percentage of ON neurons
        if p_on == 0:
            p_on = 1
        x = x / p_on  # activation correction
        return x * mask
    
    def set_mask(self, mask):
        self.masks = mask

    def disable_mask(self):
        self.masks = None


    def forward(self, x, idx=None, return_intermediates=False, layer_indexes=[], CND_reg_only_last_layer=False, apply_mask=False, por_neuron=1.0):
        """
        x: LongTensor of shape [batch_size, seq_length]
        return_intermediates: if True, return (logits, intermediates)
        """

        intermediates = [] if return_intermediates else None

        # Embed: [B, T] -> [B, T, D]
        emb = self.embedding(x).to(self.device)

        # # SOLUTION Yu19
        # out = emb.view((1, emb.size()[0], -1)) # (1, 128, 300 000)
        # out = self.avgpool(out)
        # out = out.squeeze(0)

        # Our Sol
        # Pool: [B, T, D] -> [B, D, 16]
        emb = emb.permute(0, 2, 1) #[B, D, T]
        emb = self.pool(emb)
        # Restore shape: [B, 16, D] -> [B, 20*D]
        emb = emb.permute(0, 2, 1).contiguous()
        bsz = emb.size(0)
        out = emb.view(bsz, -1)


        # # Our Sol adapted to Yu19 (pool with shape 6)
        # # Pool: [B, T, D] -> [B, D, 20]
        # emb = self.pool(emb)
        # # Restore shape: [B, 16, D] -> [B, 20*D]
        # bsz = emb.size(0)
        # out = emb.view(bsz, -1)

        if return_intermediates:
            intermediates.append(out)

        # First dense + BN + Softsign
        out = self.fc1(out)                      # [B, 4*D]
        if return_intermediates:
            intermediates.append(out)
        out = self.bn1(out)
        out = F.softsign(out)
        out = self.dropout(out) #NOT IN THE ORIGINAL VERSION


        # Second dense + Softsign
        out = self.fc2(out)                      # [B, D]
        if return_intermediates:
            intermediates.append(out)
        out = self.bn2(out)
        out = F.softsign(out)
        out = self.dropout(out) #NOT IN THE ORIGINAL VERSION

        # Final classification
        logits = self.fc3(out)                   # [B, num_classes]

        if return_intermediates:
            # Concatenate intermediates along feature dimension
            interm = torch.cat(intermediates, dim=1)
            return logits, interm
        return logits, None



class FullyConnectedNN_preact_CND(nn.Module):
    def __init__(self, device, input_channels=1, input_size=28, num_classes=10, L=3, N=128, batch_normalization_flag=False, dropout_rate=0.0, embedding_weights=None, activation_fn='relu'):
        super(FullyConnectedNN_preact_CND, self).__init__()
        self.L = L  # Number of layers
        self.N = N  # Neurons per hidden layer
        self.batch_normalization_flag = batch_normalization_flag
        self.device = device
        self.dropout_rate = dropout_rate
        self.masks = None
        self.activation_fn = activation_fn

        if embedding_weights is not None:
            self.embedding = torch.nn.Embedding.from_pretrained(embedding_weights, freeze=False)
            self.embedding_dim = self.embedding.embedding_dim
            self.news_mode = True
        else:
            self.news_mode = False

        if not self.news_mode: #IMAGES
            self.input_view = input_channels * input_size * input_size
        else:
            self.input_view  = self.embedding_dim


        # Create a list to hold the fully connected layers and batch normalization layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_normalization_flag else None
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer
            
        # First layer from input to first hidden layer
        self.layers.append(nn.Linear(self.input_view, N))
        if batch_normalization_flag:
            self.batch_norms.append(nn.BatchNorm1d(N))
            
        # Hidden layers
        for _ in range(L-1):
            self.layers.append(nn.Linear(N, N))
            if batch_normalization_flag:
                self.batch_norms.append(nn.BatchNorm1d(N))

        # Final output layer (assuming 10 output classes as in your original code)
        self.output_layer = nn.Linear(N, num_classes)

    def apply_mask(self, x, mask):
        p_on = torch.sum(mask) / mask.numel()  # percentage of ON neurons
        if p_on == 0:
            p_on = 1
        x = x / p_on  # activation correction
        return x * mask
    
    def set_mask(self, mask):
        self.masks = mask

    def disable_mask(self):
        self.masks = None

    def forward(self, x, idx, return_intermediates=False, layer_indexes=[], CND_reg_only_last_layer=False, apply_mask=False, por_neuron=1.0):
        """
        Forward pass with optional neuron masking during inference.

        Parameters:
        - x: Input tensor.
        - idx: Index (unused in this implementation, included for compatibility).
        - return_intermediates: Whether to return intermediate activations.
        - layer_indexes: List of layer indexes to include in intermediates.
        - CND_reg_only_last_layer: Whether to apply CND regularization only to the last layer.
        - masks: Dictionary of masks, where keys are layer indices and values are 1D tensors of the same size as the layer's output.

        Returns:
        - x: Final output of the model.
        - intermediates (optional): Concatenated intermediate activations.
        """
        if getattr(self, "news_mode", False):
            embeddings = self.embedding(x)  # [B, T, D]
            mask = (x != 0).float()  # [B, T], 0 correspond to <UNK>
            mask = mask.unsqueeze(-1)  # [B, T, 1]
            summed = (embeddings * mask).sum(dim=1)  # sum valid tokens
            count = mask.sum(dim=1).clamp(min=1e-9)  # avoid division by zero
            x = summed / count  # average only valid tokens
            # x = F.relu(self.fc1(pooled))
            # x = self.dropout(x)
            # x = self.output_layer(x)
            # return x, None

        x = x.view(-1, self.input_view)  # Flatten the input
        intermediates = []

        # Pass through each layer
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Apply masks if provided during inference
            if self.masks and (i in self.masks):
                x = self.apply_mask(x, self.masks[i])

            if return_intermediates:
                if (not CND_reg_only_last_layer) or (i == len(self.layers) - 1):
                    neuron_index = int(x.shape[1] * por_neuron)
                    intermediates.append(x[:, :neuron_index])

            x = F.softsign(x) if self.activation_fn == 'softsign' else F.relu(x)

            # Apply dropout
            x = self.dropout(x)

        # Final output layer (no activation)
        logits = self.output_layer(x)

        if return_intermediates:
            return logits, torch.cat(intermediates, dim=1)
            
        return logits, None
    

class FullyConnectedNN_masked(nn.Module):
    def __init__(self, device):
        super(FullyConnectedNN_masked, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(28 * 28, 512).to(device)
        self.fc2 = nn.Linear(512, 256).to(device)
        self.fc3 = nn.Linear(256, 128).to(device)
        self.fc4 = nn.Linear(128, 10).to(device)
        
        # Initialize masks for each layer (1 for active, 0 for inactive)
        self.mask_fc1 = torch.ones(512, device=device)
        self.mask_fc2 = torch.ones(256, device=device)
        self.mask_fc3 = torch.ones(128, device=device)

    def apply_mask(self, x, mask):
        return x * mask

    def forward(self, x, idx, return_intermediates=False, apply_mask=False):
        x0 = x.view(-1, 28 * 28).to(self.device)  # Flatten the image and move to device
        
        # Check if the model is in evaluation mode
        if self.training:
            # Apply masks during training
            x1 = torch.relu(self.fc1(x0))
            x2 = torch.relu(self.fc2(x1))
            x3 = torch.relu(self.fc3(x2))
        else:
            # Skip masks during evaluation
            x1 = torch.relu(self.apply_mask(self.fc1(x0), self.mask_fc1))
            x2 = torch.relu(self.apply_mask(self.fc2(x1), self.mask_fc2))
            x3 = torch.relu(self.apply_mask(self.fc3(x2), self.mask_fc3))

        x4 = self.fc4(x3)
        if return_intermediates:
            return x4, [x0, x1, x2, x3]
        return x4
