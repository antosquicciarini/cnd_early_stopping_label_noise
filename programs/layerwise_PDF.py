from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.special import digamma
import numpy as np
import os
import seaborn as sns


def visualize_layerwise_activation_distributions(
    train_loader_clean, train_loader_corrupted, model, device, args
):
    """
    Visualizes the layer-wise activation distributions for clean and corrupted samples.

    Parameters:
    - train_loader_clean (DataLoader): DataLoader containing non-corrupted (clean) samples.
    - train_loader_corrupted (DataLoader): DataLoader containing corrupted samples.
    - model (nn.Module): Neural network model being analyzed.
    - device (torch.device): Device for computation (CPU/GPU).
    - args (Namespace): Argument object with configuration details.
    """

    # Visualize for clean samples
    for batch_idx, (images, labels, indices) in enumerate(train_loader_clean):
        break

    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    indices = indices.to(device, non_blocking=True)

    outputs, pre_activations = model(
        images,
        indices,
        return_intermediates=True,
        CND_reg_only_last_layer=getattr(args, "CND_reg_only_last_layer", False),
    )
    plot_activation_density(pre_activations, range(15), args, sample_type="clean")

    # Visualize for corrupted samples
    for batch_idx, (images, labels, indices) in enumerate(train_loader_corrupted):
        break

    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    indices = indices.to(device, non_blocking=True)

    outputs, pre_activations = model(
        images,
        indices,
        return_intermediates=True,
        CND_reg_only_last_layer=getattr(args, "CND_reg_only_last_layer", False),
    )
    plot_activation_density(pre_activations, range(15), args, sample_type="corrupted")


def plot_activation_density(pre_activations, sample_indices, args, sample_type="clean"):
    """
    Plots probability density functions (PDFs) of activations for each layer.

    Parameters:
    - pre_activations (Tensor): Tensor containing activations (shape: [n_neurons, n_samples]).
    - sample_indices (list): List of sample indices to visualize.
    - args (Namespace): Argument object with layer configuration and results directory.
    - sample_type (str): Type of sample (e.g., "clean", "corrupted").
    """
    for sample_idx in sample_indices:
        plt.figure(figsize=(12, 8))

        cumulative_neurons = 0
        for layer_idx, n_neurons in args.neurs_x_hid_lyr.items():
            # Determine the range of neurons for the current layer
            layer_start = cumulative_neurons
            layer_end = cumulative_neurons + n_neurons
            cumulative_neurons = layer_end

            # Extract activations for the current layer and the specified samples
            layer_activations = pre_activations[sample_idx, layer_start:layer_end].cpu().detach().numpy()

            # # Standardize activations: subtract mean and divide by standard deviation
            # layer_mean = layer_activations.mean()
            # layer_std = layer_activations.std()
            # if layer_std > 0:
            #     layer_activations = (layer_activations - layer_mean) / layer_std
            # else:
            #     print(f"Warning: Layer {layer_idx} has zero standard deviation and was not standardized.")

            # Plot the activation density using KDE
            sns.kdeplot(
                layer_activations.flatten(),
                label=f"Layer {layer_idx}",
                fill=False,
                linewidth=2,
                alpha=0.7,
            )

        # Customize the plot
        plt.title("Layer-wise Activation Probability Density Functions", fontsize=16)
        plt.xlabel("Activation Values", fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.legend(title="Layers", fontsize=12)
        plt.grid(True)
        plt.tight_layout()

        # Save the plot to the results directory
        results_dir = f"{args.results_dir}/layerwise_PDFs"
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(f"{results_dir}/{sample_type}_idx_{sample_idx}_epoch_{args.current_epoch}_layers_activation_density.svg", format="svg")
        plt.close()