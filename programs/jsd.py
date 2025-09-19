from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.special import digamma
import torch
import numpy as np
import os
import re


import torch
from SE_approximatiors import KDE_entropy_approximation, KDE, KDE_entropy_approximation_incredible, KDE_entropy_approximation_incredible_2, KDE_entropy_approximation_fourier_transform, GaussianLike_entropy_approximation, differential_shannon_entropy_hist_approx, KDE_entropy_approximation_fourier_transform_on_grid, KDE_entropy_approximation_fft


def multi_distr_jsd(sample_list, weights):

    sample_global = torch.cat(sample_list)
    JSD = compute_shannon_entropy_bins(sample_global, weights=weights)

    # Step 1: Estimate densities for each sample using KNN
    for sample in sample_list:
        JSD -= compute_shannon_entropy_bins(sample)/len(sample_list)
    if JSD<0:
        print("WOWOWOWOW")
    # for sample in sample_list:
    #     #hist, _ = np.histogram(sample, bins=10, density=True)
    #     plt.hist(sample, bins=10, alpha=0.5)# Add labels, title, legend, and gridlines
    #     plt.xlabel("Bins")
    #     plt.ylabel("Probability Density")
    #     plt.title("Probability Density Functions for Different Samples")
    #     plt.legend()
    #     plt.grid(True)

    return JSD

def PMF_plot(PMF, args):

    results_dir = args.results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    if not os.path.exists(f"{results_dir}/PMFs_images"):
        os.makedirs(f"{results_dir}/PMFs_images")


    # Define name and path
    name = f"layer_{PMF[3]}_neuron_{PMF[4]}_epoch_{PMF[2]}"

    # Convert to numpy for plotting
    pmf_center_numpy = PMF[0].numpy()
    pmf_numpy = PMF[1].numpy()

    # Plot the 7 curves
    plt.figure(figsize=(10, 6))
    for i in range(PMF[1].shape[0]):
        plt.plot(pmf_center_numpy[i], pmf_numpy[i], label=f'Curve {i}', alpha=0.7)

    # Plot the mean curve
    mean_curve = pmf_numpy.mean(axis=0)
    plt.plot(pmf_center_numpy[0], mean_curve, label='Mean Curve', color='black', linewidth=2)

    # Customize plot
    plt.title(name)
    plt.tight_layout()
    plt.grid(True)
    plt.legend()

    # Save the figure as SVG
    save_path = os.path.join(f"{results_dir}/PMFs_images", f"{name}.svg")  # Use os.path.join for cross-platform compatibility
    plt.savefig(save_path, format='svg')
    plt.close()  # Close the plot to free up memory

    return None

def transform_neurons_dict(neurs_x_hid_lyr):
    neurs_x_hid_lyr_new = {}
    cumulative_sum = 0

    for layer, neurons in neurs_x_hid_lyr.items():
        cumulative_sum += neurons
        neurs_x_hid_lyr_new[layer] = cumulative_sum

    return neurs_x_hid_lyr_new


def compute_column_pmf_vect(tensor, max_value, min_value, args, num_bins=40, min_threshold=0):
    """
    Compute Probability Mass Functions (PMFs) for each column of the input tensor.

    Parameters:
    - tensor: Input tensor of shape (num_samples, n_neurons).
    - max_value: Tensor of maximum values for each column.
    - min_value: Tensor of minimum values for each column.
    - args: Arguments object containing configuration settings.
    - num_bins: Number of bins to use for histogram computation.
    - min_threshold: Minimum threshold for valid columns.

    Returns:
    - pmf: Tensor containing PMFs of shape (n_neurons, num_bins).
    - x_values_edges: Tensor containing bin edges for each column.
    - x_values_centers: Tensor containing bin centers for each column.
    """

    n_neurons = tensor.size(1)

    # Compute bin edges and centers
    bin_width = (max_value - min_value) / num_bins
    bin_edges = torch.linspace(0, num_bins, steps=num_bins + 1, device=tensor.device).view(1, -1) * bin_width.view(-1, 1) + min_value.view(-1, 1)
    bin_centers = (bin_edges[:, :-1] + bin_edges[:, 1:]) / 2

    # Vectorized histogram computation
    # activations: [n_neurons, num_samples]
    activations = tensor.T
    # lower and upper bin bounds: [n_neurons, num_bins]
    lower = bin_edges[:, :-1]
    upper = bin_edges[:, 1:]
    # mask shape [n_neurons, num_samples, num_bins]
    mask = (activations.unsqueeze(2) >= lower.unsqueeze(1)) & (activations.unsqueeze(2) < upper.unsqueeze(1))
    # sum over samples to get counts per bin: [n_neurons, num_bins]
    histograms = mask.sum(dim=1).float()
    # assign edges and centers directly
    x_values_edges = bin_edges
    x_values_centers = bin_centers

    # Apply thresholding to histograms and normalize to compute PMFs
    histograms = torch.clip(histograms, min=min_threshold)
    pmf = histograms / histograms.sum(dim=1, keepdim=True)

    return pmf, x_values_edges, x_values_centers

def compute_column_pmf(tensor, max_value, min_value, args, num_bins=40, min_threshold=0):
    """
    Compute Probability Mass Functions (PMFs) for each column of the input tensor.

    Parameters:
    - tensor: Input tensor of shape (num_samples, n_neurons).
    - max_value: Tensor of maximum values for each column.
    - min_value: Tensor of minimum values for each column.
    - args: Arguments object containing configuration settings.
    - num_bins: Number of bins to use for histogram computation.
    - min_threshold: Minimum threshold for valid columns.

    Returns:
    - pmf: Tensor containing PMFs of shape (n_neurons, num_bins).
    - x_values_edges: Tensor containing bin edges for each column.
    - x_values_centers: Tensor containing bin centers for each column.
    """

    n_neurons = tensor.size(1)

    # Prepare tensors for storing histograms, edges, and centers
    histograms = torch.zeros(n_neurons, num_bins)
    x_values_edges = torch.zeros(n_neurons, num_bins + 1)
    x_values_centers = torch.zeros(n_neurons, num_bins)

    # Compute bin edges and centers
    bin_width = (max_value - min_value) / num_bins
    bin_edges = torch.linspace(0, num_bins, steps=num_bins + 1, device=tensor.device).view(1, -1) * bin_width.view(-1, 1) + min_value.view(-1, 1)
    bin_centers = (bin_edges[:, :-1] + bin_edges[:, 1:]) / 2

    # Compute histograms for valid columns
    for i in range(n_neurons):
        #hist = torch.histc(tensor[:, i], bins=num_bins, min=min_value[i], max=max_value[i])
        hist = torch.histc(tensor[:, i], bins=num_bins, min=min_value[i].item(), max=max_value[i].item())
        histograms[i] = hist
        x_values_edges[i] = bin_edges[i]
        x_values_centers[i] = bin_centers[i]

    # Apply thresholding to histograms and normalize to compute PMFs
    histograms = torch.clip(histograms, min=min_threshold)
    pmf = histograms / histograms.sum(dim=1, keepdim=True)

    return pmf, x_values_edges, x_values_centers


def jsd_bins_hist(pre_activations, labels, args):
    
    n_neurons = pre_activations.shape[1]
    JSD = torch.zeros(n_neurons)
    pmf_list, pmf_center_list = [], []

    # Calculate the max, min, and bin width
    max_activation, _ = torch.max(pre_activations, dim=0)
    min_activation, _ = torch.min(pre_activations, dim=0)
    bin_width = (max_activation - min_activation) / getattr(args, "JSD_bins", 40)

    counter = 0
    SD_class_i = 0
    for cl in range(args.num_classes):
        mask = labels == cl
        if sum(mask) > 0:
            sample = pre_activations[mask]
            pmfs, x_values_edges, x_values_centers = compute_column_pmf_vect(
                sample, max_activation, min_activation, args, 
                num_bins=getattr(args, "JSD_bins", 40)
            )
            pmf_list.append(pmfs)
            pmf_center_list.append(x_values_centers)
            SD_class_i += differential_shannon_entropy_hist_approx(pmfs, bin_width)
            counter += 1
    
    if counter > 0:
        JSD -= SD_class_i / counter
    
    pmf_torch = torch.stack(pmf_list)
    pmf_center_torch = torch.stack(pmf_center_list)
    pmf_mean = torch.mean(pmf_torch, axis=0)

    JSD += differential_shannon_entropy_hist_approx(pmf_mean, bin_width)
    # Handle case where bin width is 0 (max equals min)
    zero_bin_width_mask = (bin_width == 0)
    if zero_bin_width_mask.any():
        JSD[zero_bin_width_mask] = 0  # Set JSD to 0 for these neurons

    # Ensure no negative JSD values
    nan_indices = torch.isnan(JSD).nonzero(as_tuple=True)[0]
    if nan_indices.numel() > 0:
        print("Indices of NaN values:", nan_indices)

    return JSD



def compute_column_pmf_adaptive(tensor, num_bins=40, min_threshold=0):
    """
    Compute Probability Mass Functions (PMFs) for each column of the input tensor using an adaptive bin system.
    
    Instead of using the absolute minimum and maximum values, this function uses the values 
    corresponding to the low_quantile and high_quantile (e.g., 0.01% and 99.9%) as the lower and upper bounds.
    This helps to remove extreme outliers that might otherwise skew the histogram.
    
    Parameters:
        tensor (Tensor): Input tensor of shape (num_samples, n_neurons).
        num_bins (int): Number of bins to use for the histogram.
        min_threshold (float): Minimum threshold to clip the histogram counts.
        low_quantile (float): Lower quantile fraction (default 0.0001 for 0.01%).
        high_quantile (float): Upper quantile fraction (default 0.999 for 99.9%).
        
    Returns:
        pmf (Tensor): PMFs for each column, shape (n_neurons, num_bins).
        x_values_edges (Tensor): Bin edges for each neuron, shape (n_neurons, num_bins+1).
        x_values_centers (Tensor): Bin centers for each neuron, shape (n_neurons, num_bins).
        bin_widths (Tensor): Bin width for each neuron (1D tensor of length n_neurons).
    """
    n_neurons = tensor.size(1)
    device = tensor.device

    histograms = torch.zeros(n_neurons, num_bins, device=device)
    x_values_edges = torch.zeros(n_neurons, num_bins + 1, device=device)
    x_values_centers = torch.zeros(n_neurons, num_bins, device=device)
    bin_widths = torch.zeros(n_neurons, device=device)

    for i in range(n_neurons):
        col = tensor[:, i]
        # Compute adaptive bounds using the desired quantiles:
        lower_bound = torch.min(col)
        upper_bound = torch.max(col)
        # Define bin edges and centers on a uniform grid between these adaptive bounds:
        bin_edges = torch.linspace(lower_bound, upper_bound, steps=num_bins + 1, device=device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Compute histogram for the current column.
        # Values outside [lower_bound, upper_bound] are ignored.
        hist = torch.histc(col, bins=num_bins, min=lower_bound.item(), max=upper_bound.item())
        histograms[i] = hist
        x_values_edges[i] = bin_edges
        x_values_centers[i] = bin_centers
        bin_widths[i] = bin_edges[1] - bin_edges[0]

    # Ensure that no bin count is below the minimum threshold, then normalize:
    histograms = torch.clamp(histograms, min=min_threshold)
    pmf = histograms / (histograms.sum(dim=1, keepdim=True) + 1e-12)

    return pmf, x_values_edges, x_values_centers, bin_widths



def jsd_bins_hist_adaptive(pre_activations, labels, args):
    """
    Compute the discrete Jensen-Shannon Divergence (JSD) based on histograms (PMFs)
    with an adaptive binning approach. For each neuron (column), adaptive bin edges are
    computed using quantiles (via compute_column_pmf_adaptive). These global adaptive
    bin edges are then used to compute per-class PMFs, from which the discrete JSD is calculated.
    
    Parameters:
        pre_activations (Tensor): Activation tensor of shape (batch_size, n_neurons).
        labels (Tensor): Class labels for each sample (shape: batch_size,).
        args: An argument object containing at least:
              - num_classes: number of classes.
              - JSD_bins: number of histogram bins (e.g., 40).
    
    Returns:
        JSD (Tensor): A 1D tensor (length n_neurons) containing the discrete JSD for each neuron.
    """
    device = pre_activations.device
    n_neurons = pre_activations.shape[1]
    num_bins = getattr(args, "JSD_bins", 40)
    
    # Step 1: Compute global adaptive PMFs and bin edges using all pre-activations.
    # This uses quantile-based limits (by default, 0.01% and 99.9% percentiles) to determine the
    # lower and upper bounds for each neuron's distribution.
    global_pmf, global_edges, global_centers, global_bin_widths = compute_column_pmf_adaptive(
        pre_activations, num_bins=num_bins
    )
    
    # Step 2: Compute per-class PMFs using the common, adaptive bin edges computed above.
    pmf_list = []
    SD_class_sum = torch.zeros(n_neurons, device=device)
    valid_class_count = 0
    for cl in range(args.num_classes):
        mask = labels == cl
        if mask.sum() == 0:
            continue
        sample = pre_activations[mask]  # Shape: (num_samples_class, n_neurons)
        pmfs = torch.zeros(n_neurons, num_bins, device=device)
        for i in range(n_neurons):
            # Use the adaptive bin edges computed for neuron i.
            lower_bound = global_edges[i, 0].item()
            upper_bound = global_edges[i, -1].item()
            hist = torch.histc(sample[:, i], bins=num_bins, min=lower_bound, max=upper_bound)
            hist = torch.clamp(hist, min=0)
            pmfs[i] = hist / (hist.sum() + 1e-12)
        pmf_list.append(pmfs)
        SD_class_sum += differential_shannon_entropy_hist_approx(pmfs, global_bin_widths)
        valid_class_count += 1

    if valid_class_count == 0:
        return torch.zeros(n_neurons, device=device)
    
    # Step 3: Average the class PMFs and compute the mixture PMF entropy.
    pmf_torch = torch.stack(pmf_list, dim=0)  # Shape: (num_valid_classes, n_neurons, num_bins)
    pmf_mean = pmf_torch.mean(dim=0)           # Shape: (n_neurons, num_bins)
    
    # Step 4: Compute the discrete JSD as:
    #         JSD = H(mixture PMF) - (average H(class PMF))
    JSD = differential_shannon_entropy_hist_approx(pmf_mean, global_bin_widths)
    JSD -= SD_class_sum / valid_class_count
    
    # Handle pathological cases (e.g., constant distributions)
    zero_bin_mask = (global_bin_widths == 0)
    if zero_bin_mask.any():
        JSD[zero_bin_mask] = 0

    if torch.isnan(JSD).any():
        print("Warning: NaN values detected in discrete JSD computation.")
    if (JSD < 0).any():
        print("Warning: Negative JSD values computed.")
    
    return JSD



def multi_distr_jsd_kNN(sample_dict, n_neurons, max_activation, min_activation, args):

    k = 2
    epsilon = 1e-10  # Small constant to avoid log(0)

    # Function to estimate density for each neuron separately
    def knn_density_estimation(samples):
        H_list = []
        for neur_idx in range(samples.shape[1]):
            # Fit NearestNeighbors on each neuron's activations separately
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(samples[:, neur_idx].reshape(-1, 1))
            distances, _ = nbrs.kneighbors(samples[:, neur_idx].reshape(-1, 1))
            n = distances.shape[0]
            H_list.append(np.sum(np.log(n * distances[:, -1] + epsilon)) / n)
        
        return torch.tensor(H_list)

    # Initialize JSD tensor for all neurons
    JSD = torch.zeros(n_neurons)

    # Compute individual density estimations for each class
    key_to_exclude = []
    for key, samples in sample_dict.items():
        if samples.shape[0]>5: #Compute the entropy only over classes with more than 5 data points
            SD_class_i = knn_density_estimation(samples) / len(sample_dict)
            JSD -= SD_class_i  # Accumulate over classes
        else:
            key_to_exclude.append(key)

    # Compute density estimation for all samples combined
    all_samples = torch.cat([sample for i, sample in sample_dict.items() if i not in key_to_exclude], dim=0)

    JSD += knn_density_estimation(all_samples)

    # Check if there are any negative values in JSD
    if (JSD < 0).any():
        print("Negative JSD detected")

    return JSD


def jsd_KDE(pre_activations, labels, args, return_mean = False, max_n_samples=3000, model_eval=False):
    """
    Computes the JSD regularizer across all neurons in a vectorized manner without appending to a list.

    Args:
        pre_activations (Tensor): Shape (batch_size, n_neurons).
        labels (Tensor): Class labels for each sample. Shape (batch_size,).
        args: Argument object with num_classes attribute.

    Returns:
        Tensor: Regularizer scalar value.
    """

    batch_size, n_neurons = pre_activations.shape
    pre_activations = pre_activations.T
    # to avoid memory problems
    if pre_activations.shape[1]>max_n_samples:
        pre_activations = pre_activations[:,:max_n_samples]
        labels = labels[:max_n_samples]
    
    if hasattr(args, "CND_reg_type") and "NoNorm" in args.CND_reg_type:
        pre_activations_std = pre_activations
    else:
        pre_activations_std = (pre_activations - pre_activations.mean(dim=1).unsqueeze(1))/pre_activations.std(dim=1).unsqueeze(1)
        
        
    device = pre_activations_std.device

    # Compute entropy of the mixture distribution across all neurons
    #H_M = KDE_entropy_approximation(pre_activations_std)
    #H_M = KDE_entropy_approximation(pre_activations_std)
    #H_M = KDE_entropy_approximation_fft(pre_activations_std, args)
    H_M = KDE_entropy_approximation_incredible(pre_activations_std, args)
    #H_M = KDE_entropy_approximation_fourier_transform(pre_activations_std, args)
    

    # Initialize a sum for the entropy of each class-conditional distribution
    H_C_sum = torch.zeros(n_neurons, device=device)
    num_valid_classes = 0

    # Loop over each class to compute the class-conditional entropies
    for class_idx in range(args.num_classes):
        valid_samples = pre_activations_std[:, labels == class_idx]
        
        if valid_samples.numel() == 0:
            continue  # Skip if no samples for this class

        # Sum the entropies of valid classes
        #H_C_sum += KDE_entropy_approximation(valid_samples)
        #H_C_sum += KDE_entropy_approximation_fft(valid_samples, args)
        H_C_sum += KDE_entropy_approximation_incredible(valid_samples, args)
        num_valid_classes += 1

    if num_valid_classes == 0:
        return torch.tensor(0.0, device=device)

    # Calculate average entropy over classes
    H_C = H_C_sum / num_valid_classes  # Shape (n_neurons,)

    # Compute the JSD across neurons and then average
    JSD = H_M - H_C

    # Check if the loss contains NaNs
    if torch.sum(torch.isnan(JSD))>0:
        print("wow")
        # Find indices of NaN values
        nan_indices = torch.isnan(JSD).nonzero(as_tuple=True)[0]
        # Print the indices of NaN values if any exist
        if nan_indices.numel() > 0:
            print("Indices of NaN values:", nan_indices)
        else:
            print("No NaN values detected in JSD")

    if return_mean:
        if hasattr(args, "CND_reg_type") and "linear_decr" in args.CND_reg_type:
            # Calculate start and end indices for neurons in each layer
                layer_neuron_end_indexes = torch.cumsum(torch.tensor(list(args.neurs_x_hid_lyr.values())), dim=0)
                layer_neuron_start_indexes = torch.cat((torch.tensor([0]), layer_neuron_end_indexes[:-1]))

                # Define maximum and minimum values for gamma
                max_gamma = args.CND_reg_gamma
                min_gamma = args.CND_reg_gamma / 10
                num_layers = len(args.neurs_x_hid_lyr)

                # Calculate the decrement step for gamma
                gamma_step = (max_gamma - min_gamma) / (num_layers - 1)

                # Apply the linearly decreasing gamma to the JSD values
                for i in range(num_layers):
                    gamma = max_gamma - gamma_step * i
                    JSD[layer_neuron_start_indexes[i]:layer_neuron_end_indexes[i]] *= gamma

        return JSD.mean()  # Scalar
    else:
        return JSD


def layer_wise_plot(pre_activations, sample_ii, args):
    """
    Plot all probability density functions (PDFs) of each layer_jj in a single plot window.

    Args:
        pre_activations (Tensor): Activations of the network (n_neurons, n_samples).
        sample_ii (int): Index of the sample to visualize.
        args: Argument object with neuron configuration and results directory.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))  # Create a single plot window

    cumulative_neurons = 0
    for layer_idx, n_neurons in args.neurs_x_hid_lyr.items():
        # Extract activations for the current layer
        layer_start = cumulative_neurons
        layer_end = cumulative_neurons + n_neurons
        cumulative_neurons = layer_end

        layer_jj = pre_activations[layer_start:layer_end, sample_ii].cpu().detach().numpy()

        # Plot PDF for the current layer using KDE
        sns.kdeplot(
            layer_jj, 
            label=f"Layer {layer_idx}", 
            fill=False, 
            linewidth=2, 
            alpha=0.7
        )

    # Customize the plot
    plt.title("Probability Density Functions for All Layers", fontsize=16)
    plt.xlabel("Activation Values", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(title="Layers", fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to results directory
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(f"temporary_data/good_layers_pdf_{sample_ii}.svg", format="svg")
    plt.close()

def jsd_trapezoidal_kl_adaptive(pre_activations, labels, metric, args, return_mean=False, model_eval=False):
    """
    Estimate the Jensen-Shannon Divergence (JSD) using density functions with KL divergence,
    but now on an *adaptive*, non-uniform grid derived from the empirical distribution.

    Args:
        pre_activations (Tensor): Shape (batch_size, n_neurons).
        labels (Tensor): Class labels for each sample. Shape (batch_size,).
        args: Argument object with `num_classes` and optional attributes like `CND_reg_type`.
        return_mean (bool): If True, return the mean JSD value. Default is False.

    Returns:
        Tensor: JSD regularizer (either scalar or per-neuron values).
    """
    device = pre_activations.device
    batch_size, n_neurons = pre_activations.shape

    # ------------------------------------------------------------
    # 1) Standardize activations if needed
    # ------------------------------------------------------------
    # Shape: (n_neurons, batch_size) for easier neuron-wise ops
    activations_t = pre_activations.T

    if hasattr(args, "CND_reg_type") and "NoNorm" in args.CND_reg_type:
        standardized_activations = activations_t
    else:
        mean_vals = activations_t.mean(dim=1, keepdim=True)
        std_vals = activations_t.std(dim=1, keepdim=True).clamp(min=1e-5)  # avoid div-by-zero
        standardized_activations = (activations_t - mean_vals) / std_vals

    # ------------------------------------------------------------
    # 2) Prepare density functions for each class
    #    (KDE or whatever method you already have returning df)
    # ------------------------------------------------------------
    density_functions = []
    valid_class_count = 0
    for class_idx in range(args.num_classes):
        class_samples = standardized_activations[:, labels == class_idx]
        if class_samples.numel() == 0:
            continue
        # Build a PDF function for each class
        df = KDE(
            class_samples,
            args,
            return_f_est=True
        )
        density_functions.append(df)
        valid_class_count += 1

    if valid_class_count == 0:
        return torch.tensor(0.0, device=device)

    # ------------------------------------------------------------
    # 3) Build an *adaptive* grid from the entire distribution
    #    e.g. pick 64 or 128 points from sorted data as quantiles
    # ------------------------------------------------------------
    quantile_grid_size = 2**7  # or 2**6, up to you
    all_data = standardized_activations.flatten()  # (n_neurons * batch_size,)
    sorted_data = torch.sort(all_data)[0]

    # Create quantile-based indices
    if sorted_data.numel() <= quantile_grid_size:
        # if fewer points than desired grid size, just use them all
        x_adaptive = sorted_data.unsqueeze(0)  # shape: (1, N)
    else:
        indices = torch.linspace(0, sorted_data.numel() - 1, steps=quantile_grid_size)
        indices = indices.round().long().clamp(max=sorted_data.numel() - 1)
        x_adaptive = sorted_data[indices].unsqueeze(0)  # shape: (1, quantile_grid_size)
    # ------------------------------------------------------------
    # 4) Evaluate each class PDF on the new adaptive grid
    #    We'll do shape: (valid_class_count, n_neurons, #gridpts)
    # ------------------------------------------------------------
    # Typically, df(x) expects shape (1, #points), so we do df(x_adaptive).
    # The result might be shape (n_neurons, #gridpts). We'll stack them.
    densities_list = []
    for df in density_functions:
        dens = df(x_adaptive)  # Evaluate each neuron's PDF at x_adaptive
        # df(...) returns shape (n_neurons, #gridpts) presumably
        densities_list.append(dens)
    # Stack => shape: (valid_class_count, n_neurons, #gridpts)
    densities = torch.stack(densities_list, dim=0)

    # ------------------------------------------------------------
    # 5) Trapezoid-based normalization on NON-UNIFORM spacing
    #    We must do this "by hand" for each adjacent pair
    # ------------------------------------------------------------
    # x_adaptive is shape (1, #gridpts)
    # We'll compute the integral = sum_{i=0..n-2} 0.5*(pdf[i]+pdf[i+1])*(x[i+1]-x[i])
    x_vals = x_adaptive[0]  # shape (#gridpts,)
    # We'll build "area" for each class & neuron
    # shape for partial sums => (valid_class_count, n_neurons)
    total_area = torch.zeros(densities.shape[:2], device=device)

    for i in range(x_vals.shape[0] - 1):
        dx = (x_vals[i+1] - x_vals[i])
        # densities[:, :, i] => shape (valid_class_count, n_neurons)
        # average heights => 0.5*(densities[..., i] + densities[..., i+1])
        avg_height = 0.5 * (densities[:, :, i] + densities[:, :, i+1])
        total_area += avg_height * dx

    # Now normalize each curve => so each density integrates to ~1
    # shape of total_area => (valid_class_count, n_neurons)
    # We'll broadcast along the last dimension
    densities /= (total_area.unsqueeze(-1) + 1e-12)

    # ------------------------------------------------------------
    # 6) Compute the mean density across classes (shape: (n_neurons, #gridpts))
    # ------------------------------------------------------------
    mean_density = densities.mean(dim=0)  # average over first axis => (n_neurons, #gridpts)

    # ------------------------------------------------------------
    # 7) Compute the KL part of JSD
    #    JSD = (1 / #classes) * sum_{i} KL(densities[i] || M)  plus symmetrical part
    #    Actually we do JSD = sum p_i log(p_i / m_i) dx, etc.
    # ------------------------------------------------------------
    # We'll do the integral of p_i log(p_i/m) for each class i, then average
    # shape => JSD per neuron
    jsd_per_neuron = torch.zeros(n_neurons, device=device)

    # We'll do numeric integral for each class, each neuron, over the non-uniform grid
    for c_idx in range(valid_class_count):
        # p_i = densities[c_idx, neuron, :]
        # m   = mean_density[neuron, :]
        # integral => sum_{bins} p_i[bins]*log(p_i[bins]/m[bins]) * dx
        # again, non-uniform trapezoid for the function p_i log(p_i/m)
        class_density = densities[c_idx]  # shape (n_neurons, #gridpts)

        for i in range(x_vals.shape[0] - 1):
            dx = x_vals[i+1] - x_vals[i]
            # average height for each neuron:
            p_left  = class_density[:, i].clamp(min=1e-10)
            p_right = class_density[:, i+1].clamp(min=1e-10)
            m_left  = mean_density[:, i].clamp(min=1e-10)
            m_right = mean_density[:, i+1].clamp(min=1e-10)

            # Evaluate the function = p * log(p/m) at left and right
            val_left  = p_left  * torch.log(p_left  / m_left)
            val_right = p_right * torch.log(p_right / m_right)
            # Trapezoid
            avg_height = 0.5 * (val_left + val_right)
            jsd_per_neuron += avg_height * dx

    # Average across classes
    jsd_per_neuron /= valid_class_count

    # Plots
    generate_PDF_plots(x_vals, densities, mean_density, metric, jsd_per_neuron, mean_vals, std_vals, args)

    # ------------------------------------------------------------
    # 8) Optional layer-wise scaling or "linear_decr" logic
    # ------------------------------------------------------------
    if return_mean:
        if hasattr(args, "CND_reg_type") and "linear_decr" in args.CND_reg_type:
            layer_neuron_ends = torch.cumsum(
                torch.tensor(list(args.neurs_x_hid_lyr.values()), device=device), dim=0
            )
            layer_neuron_starts = torch.cat((torch.tensor([0], device=device), layer_neuron_ends[:-1]))
            max_gamma = args.CND_reg_gamma
            min_gamma = args.CND_reg_gamma / 10
            num_layers = len(args.neurs_x_hid_lyr)
            gamma_step = (max_gamma - min_gamma) / (num_layers - 1)

            for i in range(num_layers):
                gamma = max_gamma - gamma_step * i
                s_idx = layer_neuron_starts[i].item()
                e_idx = layer_neuron_ends[i].item()
                jsd_per_neuron[s_idx:e_idx] *= gamma

        return jsd_per_neuron.mean()
    else:
        return jsd_per_neuron

def jsd_trapezoidal_kl_adaptive_chebyshev(pre_activations, labels, metric, args, return_mean=False, model_eval=False):
    """
    Estimate the Jensen-Shannon Divergence (JSD) using density functions with KL divergence,
    but now on an *adaptive*, non-uniform grid derived from the empirical distribution using Chebyshev points.

    Args:
        pre_activations (Tensor): Shape (batch_size, n_neurons).
        labels (Tensor): Class labels for each sample. Shape (batch_size,).
        args: Argument object with `num_classes` and optional attributes like `CND_reg_type`.
        return_mean (bool): If True, return the mean JSD value. Default is False.
        model_eval (bool): Flag for model evaluation (default False).

    Returns:
        Tensor: JSD regularizer (either scalar or per-neuron values).
    """

    device = pre_activations.device
    batch_size, n_neurons = pre_activations.shape

    # ------------------------------------------------------------
    # 1) Standardize activations if needed
    # ------------------------------------------------------------
    # Transpose to shape: (n_neurons, batch_size)
    activations_t = pre_activations.T
    if hasattr(args, "CND_reg_type") and "NoNorm" in args.CND_reg_type:
        standardized_activations = activations_t
    else:
        mean_vals = activations_t.mean(dim=1, keepdim=True)
        std_vals = activations_t.std(dim=1, keepdim=True).clamp(min=1e-5)  # avoid div-by-zero
        standardized_activations = (activations_t - mean_vals) / std_vals

    # ------------------------------------------------------------
    # 2) Prepare density functions for each class
    # ------------------------------------------------------------
    density_functions = []
    valid_class_count = 0
    for class_idx in range(args.num_classes):
        class_samples = standardized_activations[:, labels == class_idx]

        if class_samples.numel() == 0:
            continue
        # Build a PDF function for each class using KDE (or your preferred method)
        df = KDE(class_samples, args, return_f_est=True)
        density_functions.append(df)
        valid_class_count += 1

    if valid_class_count == 0:
        return torch.tensor(0.0, device=device)

    # ------------------------------------------------------------
    # 3) Build an adaptive grid using Chebyshev points
    # ------------------------------------------------------------
    desired_grid_size = 2**8  # For example, 256 points

    # With the NEWS dataset, the number of neurons to analyse its too high for this code, thus we limit for that specific case the numer of samples to be analysed
    if args.network == "NewsMLP":
        standardized_activations = standardized_activations[:,:500]
        
    all_data = standardized_activations.flatten()  # shape: (n_neurons * batch_size,)

    sorted_data = torch.sort(all_data)[0]

    if sorted_data.numel() <= desired_grid_size:
        # If there are fewer points than desired, just use them all.
        x_adaptive = sorted_data.unsqueeze(0)  # shape: (1, N)
    else:
        # Define quantile-based endpoints to reduce the impact of extreme tail values.
        a = torch.quantile(sorted_data, 0.0001)
        b = torch.quantile(sorted_data, 0.9999)
        N = desired_grid_size  # number of Chebyshev nodes
        k = torch.arange(0, N, device=device, dtype=torch.float32)
        # Compute Chebyshev nodes of the second kind on [a, b]:
        #   x_k = 0.5*(a+b) + 0.5*(b-a)*cos(pi*k/(N-1))
        x_cheb = 0.5 * (a + b) + 0.5 * (b - a) * torch.cos(torch.pi * k / (N - 1))
        # The nodes are in descending order (from b to a); reverse to obtain increasing order.
        x_adaptive = torch.flip(x_cheb, dims=[0]).unsqueeze(0)  # shape: (1, N)

    # ------------------------------------------------------------
    # 4) Evaluate each class PDF on the new adaptive grid
    # ------------------------------------------------------------
    densities_list = []
    for df in density_functions:
        dens = df(x_adaptive)  # Expected shape: (n_neurons, #gridpts)
        densities_list.append(dens)
    # Stack the densities: shape (valid_class_count, n_neurons, #gridpts)
    densities = torch.stack(densities_list, dim=0)

    # ------------------------------------------------------------
    # 5) Trapezoidal-based normalization on non-uniform spacing
    # ------------------------------------------------------------
    x_vals = x_adaptive[0]  # shape: (#gridpts,)
    total_area = torch.zeros(densities.shape[:2], device=device)
    for i in range(x_vals.shape[0] - 1):
        dx = (x_vals[i+1] - x_vals[i])
        # Average density values at adjacent grid points
        avg_height = 0.5 * (densities[:, :, i] + densities[:, :, i+1])
        total_area += avg_height * dx
    # Normalize each density curve so that it integrates to 1
    densities /= (total_area.unsqueeze(-1) + 1e-12)

    # ------------------------------------------------------------
    # 6) Compute the mean density across classes
    # ------------------------------------------------------------
    mean_density = densities.mean(dim=0)  # shape: (n_neurons, #gridpts)

    # ------------------------------------------------------------
    # 7) Compute the KL divergence part of JSD using trapezoidal integration
    # ------------------------------------------------------------
    jsd_per_neuron = torch.zeros(n_neurons, device=device)
    for c_idx in range(valid_class_count):
        class_density = densities[c_idx]  # shape: (n_neurons, #gridpts)
        for i in range(x_vals.shape[0] - 1):
            dx = x_vals[i+1] - x_vals[i]
            # Ensure numerical stability by clamping values
            p_left  = class_density[:, i].clamp(min=1e-10)
            p_right = class_density[:, i+1].clamp(min=1e-10)
            m_left  = mean_density[:, i].clamp(min=1e-10)
            m_right = mean_density[:, i+1].clamp(min=1e-10)
            # Compute the integrand p * log(p/m) at the left and right points
            val_left  = p_left  * torch.log(p_left  / m_left)
            val_right = p_right * torch.log(p_right / m_right)
            avg_height = 0.5 * (val_left + val_right)
            jsd_per_neuron += avg_height * dx
    jsd_per_neuron /= valid_class_count

    # ------------------------------------------------------------
    # 7) (Optional) Generate PDF plots
    # ------------------------------------------------------------
    generate_PDF_plots(x_vals, densities, mean_density, metric, jsd_per_neuron, mean_vals, std_vals, args)

    # ------------------------------------------------------------
    # 8) Optional layer-wise scaling (linear_decr) logic
    # ------------------------------------------------------------
    if return_mean:
        if hasattr(args, "CND_reg_type") and "linear_decr" in args.CND_reg_type:
            layer_neuron_ends = torch.cumsum(
                torch.tensor(list(args.neurs_x_hid_lyr.values()), device=device), dim=0
            )
            layer_neuron_starts = torch.cat((torch.tensor([0], device=device), layer_neuron_ends[:-1]))
            max_gamma = args.CND_reg_gamma
            min_gamma = args.CND_reg_gamma / 10
            num_layers = len(args.neurs_x_hid_lyr)
            gamma_step = (max_gamma - min_gamma) / (num_layers - 1)
            for i in range(num_layers):
                gamma = max_gamma - gamma_step * i
                s_idx = layer_neuron_starts[i].item()
                e_idx = layer_neuron_ends[i].item()
                jsd_per_neuron[s_idx:e_idx] *= gamma

        return jsd_per_neuron.mean()
    else:
        return jsd_per_neuron
    



def jsd_trapezoidal_kl(pre_activations, labels, metric, args, return_mean=False, model_eval=False):
    """
    Estimate the Jensen-Shannon Divergence (JSD) using density functions with KL divergence,
    based on the trapezoidal integral approximation.

    Args:
        pre_activations (Tensor): Shape (batch_size, n_neurons).
        labels (Tensor): Class labels for each sample. Shape (batch_size,).
        args: Argument object with `num_classes` and optional attributes like `CND_reg_type`.
        return_mean (bool): If True, return the mean JSD value. Default is False.
        max_n_samples (int): Maximum number of samples to consider to avoid memory issues.

    Returns:
        Tensor: JSD regularizer (either scalar or per-neuron values).
    """
    # Step 1: Prepare activations and handle memory constraints
    batch_size, n_neurons = pre_activations.shape
    pre_activations = pre_activations.T  # Transpose to shape (n_neurons, batch_size)
    
    # Step 2: Standardize activations if needed
    if hasattr(args, "CND_reg_type") and "NoNorm" in args.CND_reg_type:
        standardized_activations = pre_activations
    else:
        mean = pre_activations.mean(dim=1, keepdim=True)
        std = pre_activations.std(dim=1, keepdim=True).clamp(min=1e-5)  # Avoid division by zero
        standardized_activations = (pre_activations - mean) / std

    device = standardized_activations.device

    # Step 3: Compute densities for each class
    density_functions = []
    valid_class_count = 0
    for class_idx in range(args.num_classes):
        class_samples = standardized_activations[:, labels == class_idx]
        if class_samples.numel() == 0:
            continue

        density_function = KDE( #KDE_entropy_approximation_fourier_transform(  #KDE( 
            class_samples, args, return_f_est=True
        )
        density_functions.append(density_function)
        valid_class_count += 1

    if valid_class_count == 0:
        return torch.tensor(0.0, device=device)

    # Step 4: Compute densities across all classes on a discrete grid
    x = torch.linspace(-5, 5, 2**6, device=device).unsqueeze(0)
    # Step 5: Calculate trapezoidal bin widths
    bin_widths = x[:, 1:] - x[:, :-1]
    densities = torch.stack([df(x) for df in density_functions])

    # Normalize densities
    density_sums = torch.sum(densities[:, :, :-1] * bin_widths, dim=2)
    densities = densities / density_sums.unsqueeze(2)

    mean_density = densities.mean(dim=0)

    densities = densities.clamp(min=1e-10)
    mean_density = mean_density.clamp(min=1e-10)

    # Step 6: Compute KL divergence for each class
    jsd = torch.zeros(n_neurons, device=device)
    for i in range(valid_class_count):
        jsd += torch.sum(
            densities[i, :, :-1]
            * (torch.log(densities[i, :, :-1]) - torch.log(mean_density[:, :-1]))
            * bin_widths,
            dim=1,
        )

    # Step 7: Calculate final JSD
    JSD = jsd / valid_class_count

    if torch.sum(torch.isnan(pre_activations)) >0:
        print("Loss became NaN. Terminating training.")
        raise ValueError("Loss is NaN. Stopping training.")
    
    # 8) Plots
    generate_PDF_plots(x, densities, mean_density, metric, JSD, args)
                
    # Step 9: Apply optional layer-wise scaling and return
    if return_mean:
        if hasattr(args, "CND_reg_type") and "linear_decr" in args.CND_reg_type:
            layer_neuron_end_indexes = torch.cumsum(
                torch.tensor(list(args.neurs_x_hid_lyr.values()))
            )
            layer_neuron_start_indexes = torch.cat(
                (torch.tensor([0]), layer_neuron_end_indexes[:-1])
            )
            max_gamma = args.CND_reg_gamma
            min_gamma = args.CND_reg_gamma / 10
            num_layers = len(args.neurs_x_hid_lyr)
            gamma_step = (max_gamma - min_gamma) / (num_layers - 1)

            for i in range(num_layers):
                gamma = max_gamma - gamma_step * i
                JSD[layer_neuron_start_indexes[i] : layer_neuron_end_indexes[i]] *= gamma

        return JSD.mean()
    else:
        return JSD

def generate_PDF_plots(x, densities, mean_density, metric, JSD, mean, std, args):
    por_neuron = getattr(args, "por_neurons_x_layer_preact", 1.)
    
    for key, value in args.neurs_x_hid_lyr.items():
        if args.current_epoch % 10 == 0:
            # Get the range of neuron indices for the current layer
            layer_start = sum([args.neurs_x_hid_lyr[k] for k in range(key)]) if key > 0 else 0
            layer_end = layer_start + value
            layer_jsd = JSD[layer_start:layer_end]  # JSD for the current layer

            # Determine the number of neurons to plot (up to 4)
            num_neurons_to_plot = min(4, layer_jsd.size(0))

            # Select the first 4 neuron indices from the current layer
            selected_indices = torch.arange(0, num_neurons_to_plot, device=layer_jsd.device)

            # Squeeze mean and std
            mean = mean.squeeze()
            std = std.squeeze()

            # Plot for randomly selected neurons
            for idx in selected_indices:
                neuron_index = layer_start + idx
                PDF_plot(
                    x,
                    densities[:, neuron_index, :],
                    mean_density[neuron_index, :],
                    args.current_epoch,
                    key,
                    neuron_index,
                    metric,
                    args,
                    mean=mean[neuron_index],
                    std=std[neuron_index],
                    quartile="random",
                )
    return None

    
def PDF_plot(x, densities, mean_density, epoch, layer_index, metric, neuron_index, args, mean=0.0, std=1.0, quartile="low"):
    """
    Plot the Probability Density Function (PDF) for given densities and save the plot.

    Parameters:
    - x: Tensor of x-axis values.
    - densities: Tensor of densities corresponding to x-axis values.
    - mean_density: Tensor of the mean density for the neuron.
    - epoch: Current epoch number.
    - layer_index: Index of the layer being plotted.
    - neuron_index: Index of the neuron being plotted.
    - args: Arguments object containing configuration settings.
    - mean: Mean of the data for the current neuron (default=0.0).
    - std: Standard deviation of the data for the current neuron (default=1.0).
    - quartile: A string to indicate whether the neuron has a "low" or "high" CND.

    Returns:
    - None
    """
    # Adjust x-axis values based on the provided mean and standard deviation
    x_scaled = x.squeeze() * std + mean

    # Normalize densities so that their trapezoidal integral equals 1
    bin_widths = x_scaled[1:] - x_scaled[:-1]
    densities = densities / torch.sum(densities[:, :-1] * bin_widths, dim=1, keepdim=True)
    mean_density = mean_density / torch.sum(mean_density[:-1] * bin_widths)

    # Define the results directory and ensure it exists
    results_dir = args.results_dir
    images_dir = os.path.join(results_dir, "PMFs_images")
    os.makedirs(images_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Define the file name and path for saving the plot
    plot_name = f"{metric}_{quartile}_epoch_{epoch}_neuron_{neuron_index}_CND.svg"
    save_path = os.path.join(images_dir, plot_name)

    # Create the plot with updated figure size and resolution
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(x_scaled.cpu(), densities.T.cpu(), linewidth=1.0, alpha=0.8)
    plt.plot(x_scaled.cpu(), mean_density.cpu(), label="Mean Density", linewidth=2.0, color="black")

    # Customize the plot with updated font sizes
    #plt.title(f"PDF - {plot_name}", fontsize=16)
    #plt.xlabel("Pre-Activation Values", fontsize=16)
    #plt.ylabel("Class-conditioned PDFs", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #plt.xlim(-2, 7)
    plt.grid(True)
    #plt.legend(fontsize=16)
    plt.tight_layout()

    # Save the plot and clean up
    plt.savefig(save_path, format="svg")
    plt.close()
    return None

def jsd_GaussianLike(pre_activations, labels, args, return_mean = False):
    """
    Computes the JSD regularizer across all neurons in a vectorized manner without appending to a list.

    Args:
        pre_activations (Tensor): Shape (batch_size, n_neurons).
        labels (Tensor): Class labels for each sample. Shape (batch_size,).
        args: Argument object with num_classes attribute.

    Returns:
        Tensor: Regularizer scalar value.
    """
    batch_size, n_neurons = pre_activations.shape
    pre_activations = pre_activations.T
    device = pre_activations.device

    # Compute entropy of the mixture distribution across all neurons
    H_M = GaussianLike_entropy_approximation(pre_activations)

    # Initialize a sum for the entropy of each class-conditional distribution
    H_C_sum = torch.zeros(n_neurons, device=device)
    num_valid_classes = 0

    # Loop over each class to compute the class-conditional entropies
    for class_idx in range(args.num_classes):
        valid_samples = pre_activations[:, labels == class_idx]
        
        if valid_samples.numel() == 0:
            continue  # Skip if no samples for this class

        # Sum the entropies of valid classes
        H_C_sum += GaussianLike_entropy_approximation(valid_samples)
        num_valid_classes += 1

    if num_valid_classes == 0:
        return torch.tensor(0.0, device=device)

    # Calculate average entropy over classes
    H_C = H_C_sum / num_valid_classes  # Shape (n_neurons,)

    # Compute the JSD across neurons and then average
    JSD = H_M - H_C
    if return_mean:
        return JSD.mean()  # Scalar
    else:
        return JSD


def mean_approx_JSD(sample_dict, n_neurons, max_activation, args):
    # Create a list to store the mean for each tensor in sample_dict
    mean_list = []
    
    for key, values in sample_dict.items():
        # Apply the mask where values are greater than args.CND_min_th
        mask = values > args.CND_min_th
        
        # Count the number of valid entries along axis 0
        valid_counts = mask.sum(dim=0)
        
        # Replace invalid values with 0
        filtered_values = values * mask
        
        # Compute the sum of valid values
        sum_filtered = filtered_values.sum(dim=0)
        
        # Compute the mean by dividing by the number of valid entries
        mean_filtered = sum_filtered / valid_counts
        
        # Handle division by zero, set to 0 if no valid entries exist for a column
        mean_filtered[valid_counts == 0] = 0
        
        # Append the mean for the current values to the list
        mean_list.append(mean_filtered)
    
    # Stack the means from all sample_dict entries and compute the final mean across all
    CDN_mean_scores = torch.mean(torch.stack(mean_list), dim=0)
    
    return CDN_mean_scores



def topK_jsd_bins_hist_approx(sample_dict, n_neurons, max_activation, min_activation, args, CND_type="CND_PMF_top1"):

    match = re.search(r"top([1-9])", CND_type)
    if match:
        k = int(match.group(1))
    else:
        raise ValueError("No valid 'top' number (1-9) found in the text.")

    def discrete_shannon_entropy(pmf, bin_width):
        mask_1 = (torch.max(pmf, axis=1)[0] - torch.min(pmf, axis=1)[0]) == 0 # To detect constant distributions and do not make them contributing to the final JSD
        mask_2 = (bin_width == 0) # Detect one-hot distributions
        mask = mask_1 | mask_2
        discrete_shannon_entropy = -torch.sum(pmf * torch.log2(pmf / bin_width.unsqueeze(1)), axis=1)

        discrete_shannon_entropy[mask] = 0.0 #not consider the effect of uniform distributions
        return discrete_shannon_entropy

    pmf_list, pmf_center_list = [], []
    for key, sample in sample_dict.items():
        pmfs, x_values_edges, x_values_centers = compute_column_pmf(sample, max_activation, min_activation, args, num_bins=args.JSD_bins, min_threshold=args.CND_min_th)
        pmf_list.append(pmfs)
        pmf_center_list.append(x_values_centers)
    pmf_torch = torch.stack(pmf_list)
    pmf_center_torch = torch.stack(pmf_center_list)

    bin_width = (max_activation - min_activation)/args.JSD_bins

    JSD_list = []
    for ii in range(pmf_torch.shape[0]):

        # Exclude the i-th slice along the first dimension
        pmf_excluded = torch.cat((pmf_torch[:ii], pmf_torch[ii+1:]), dim=0)
        # Compute the mean along the first dimension
        pmf_mean = torch.mean(pmf_excluded, dim=0)

        JSD_list.append(
            discrete_shannon_entropy(torch.mean(torch.stack((pmf_mean, pmf_torch[ii])), dim=0), bin_width) + \
        - 1/2*discrete_shannon_entropy(pmf_mean, bin_width) - 1/2*discrete_shannon_entropy(pmf_torch[ii], bin_width)
        )

    JSD_torch = torch.stack(JSD_list) 
    top_k_values, _ = torch.topk(JSD_torch, k, dim=0)
    JSD_top_k = torch.mean(top_k_values, dim=0)
        
    if hasattr(args, 'PMFs_epochs_to_plot') and args.current_epoch in args.PMFs_epochs_to_plot:
        neuron_x_layer_dict = transform_neurons_dict(args.neurs_x_hid_lyr)

        for key, value in neuron_x_layer_dict.items():
            # Save two neurons x layer
            PMF_plot((pmf_center_torch[:,value-2,:], pmf_torch[:,value-2,:], args.current_epoch, key, value-2), args)
            PMF_plot((pmf_center_torch[:,value-3,:], pmf_torch[:,value-3,:], args.current_epoch, key, value-3), args)
            PMF_plot((pmf_center_torch[:,value-4,:], pmf_torch[:,value-4,:], args.current_epoch, key, value-4), args)
            PMF_plot((pmf_center_torch[:,value-5,:], pmf_torch[:,value-5,:], args.current_epoch, key, value-5), args)

    return JSD_top_k


def nearest_neighbor_entropy(X):
  """
  Estimates differential entropy using the nearest neighbor method.

  Args:
    X: A PyTorch tensor of shape (N, d), where N is the number of data points and d is the dimensionality.

  Returns:
    The estimated differential entropy.
  """

  N, d = X.shape
  distances = torch.cdist(X, X, p=2)  # Euclidean distances
  nearest_neighbor_distances = torch.min(distances, dim=1, keepdim=False)[0]

  # Calculate the entropy estimate
  entropy = (1 / N) * torch.sum(torch.log(nearest_neighbor_distances * N) + torch.log(2) + np.euler_gamma)

  return entropy








# Step 1: Generate i.i.d. samples from two distributions (p(x) and q(x))
def generate_samples(n, m, p_dist, q_dist):
    """
    Generate n samples from p(x) and m samples from q(x).
    p_dist and q_dist are callable distributions from PyTorch.
    """
    X_p = torch.sort(p_dist(n))[0]  # n samples from p(x)
    X_q = torch.sort(q_dist(m))[0]  # m samples from q(x)
    return X_p, X_q

# Step 2: Empirical CDF estimator
def empirical_cdf(samples):
    """
    Compute the empirical CDF from a sample set using PyTorch.
    Returns a piecewise linear extension of the empirical CDF.
    """
    n = len(samples)
    cdf_values = torch.arange(1, n + 1, dtype=torch.float32) / n  # Values of the CDF at the sample points
    return lambda x: torch.interp(x, samples, cdf_values, left=0.0, right=1.0)

# Step 3: KL divergence estimation
def kl_divergence(X_p, X_q, eps=1e-5):
    """
    Estimate the KL divergence between two distributions given i.i.d. samples using PyTorch.
    """
    n = len(X_p)
    
    # Compute empirical CDFs
    P_c = empirical_cdf(X_p)
    Q_c = empirical_cdf(X_q)
    
    # KL divergence estimation using the CDFs
    kl_sum = 0.0
    for i in range(n):
        x_i = X_p[i]
        # Calculate delta P_c and Q_c with a small epsilon
        delta_P_c = P_c(x_i) - P_c(x_i - eps)
        delta_Q_c = Q_c(x_i) - Q_c(x_i - eps)
        
        # Avoid division by zero or log(0) errors
        if delta_P_c > 0 and delta_Q_c > 0:
            kl_sum += torch.log(delta_P_c / delta_Q_c)
    
    kl_div = kl_sum / n
    return kl_div




#Old codes

def kl_divergence(p, q):
    p = np.clip(p, 1e-10, None)
    q = np.clip(q, 1e-10, None)
    return np.sum(p * np.log(p / q))

def knn_density_estimation(samples, k):
    samples = samples.cpu().detach().numpy()
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(samples.reshape(-1, 1))
    distances, _ = nbrs.kneighbors(samples.reshape(-1, 1))
    density_estimates = k / (distances[:, -1])  # No need for extra power since it's 1D #k / (np.mean(distances[:, -1]) ** samples.shape[1])
    return density_estimates

def kl_divergence(p, q):
    p = np.clip(p, 1e-10, None)
    q = np.clip(q, 1e-10, None)
    return np.sum(p * np.log(p / q))

def knn_density_estimation(samples, k):
    samples = samples.cpu().detach().numpy()
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(samples.reshape(-1, 1))
    distances, _ = nbrs.kneighbors(samples.reshape(-1, 1))
    density_estimates = k / (distances[:, -1])  # No need for extra power since it's 1D #k / (np.mean(distances[:, -1]) ** samples.shape[1])
    return density_estimates


def kl_divergence_estimation(p_data, q_data):
    """
    Estimate the KL divergence between two distributions P and Q
    using the empirical and continuous piecewise CDFs.
    """
    eps = 1e-6 #check the min
    p_data_sorted = np.sort(p_data)
    q_data_sorted = np.sort(q_data)

    n = len(p_data)
    n0 = np.sum(p_data_sorted==0) #len(p_data_sorted)

    kl_divergence = 0.0
    for i in range(n0, n):
        x = p_data_sorted[i]

        # Continuous piecewise CDFs for P and Q
        P_c_x = continuous_piecewise_cdf(p_data_sorted, x)
        Q_c_x = continuous_piecewise_cdf(q_data_sorted, x)

        # Delta CDF values (difference between CDF at x and a point just before x)
        if i == 0:
            delta_P_c = P_c_x
        else:
            delta_P_c = P_c_x - continuous_piecewise_cdf(p_data_sorted, x - eps) #check this
            delta_Q_c = Q_c_x - continuous_piecewise_cdf(q_data_sorted, x - eps) #check this
        
        if delta_P_c > 0 and Q_c_x > 0:
            kl_divergence += np.log(delta_P_c / delta_Q_c) 

    kl_divergence /= n
    return kl_divergence


# # Example usage
# N = 3  # Number of sample sets
# k = 5  # Number of nearest neighbors

# # Assume we have N sets of samples
# samples1 = np.random.normal(0, 1, 100)
# samples2 = np.random.normal(1, 1, 100)
# samples3 = np.random.normal(2, 1, 100)

# # Create a list of sample sets
# sample_list = [samples1, samples2, samples3]

# # Compute JSD for N samples
# jsd = multi_sample_jsd(sample_list, k)
# print(f"Jensen-Shannon Divergence for {N} samples: {jsd}")

def empirical_cdf(data, x):
    """
    Compute the empirical CDF at point x for the given data.
    """
    return np.sum((data <= x) & (data != 0)) / np.sum(data!=0)

def continuous_piecewise_cdf(data, x):
    """
    Compute the continuous piecewise linear CDF extension at point x.
    """
    n = len(data)
    if x < data[0]:
        return 0.0
    if x >= data[-1]:
        return 1.0
    
    for i in range(1, n):
        if data[i-1] <= x < data[i]:
            a = (i - 1) / n
            b = (i) / n
            slope = (b - a) / (data[i] - data[i-1])
            return a + slope * (x - data[i-1])
    
    return 1.0


def piecewise_cdf_and_pdf(samples, num_points=1000):
    """
    Computes the continuous piecewise CDF and its derivative (PDF) from empirical samples.

    Parameters:
    - samples: array-like, the empirical samples.
    - num_points: int, the number of points to use for interpolation in the CDF.

    Returns:
    - x: array, the x values for the CDF and PDF.
    - cdf: array, the corresponding CDF values.
    - pdf: array, the corresponding PDF values (derivative of CDF).
    """
    # Filter out zero values from samples
    samples = samples[samples != 0]
    
    # Check if samples array is empty after filtering
    if len(samples) == 0:
        raise ValueError("No valid samples to compute CDF and PDF after filtering zeros.")

    # Sort the samples
    sorted_samples = np.sort(samples)
    n = len(sorted_samples)

    # Compute the ECDF
    ecdf_y = np.arange(1, n + 1) / n

    # Create points for the CDF
    x = np.linspace(np.min(sorted_samples), np.max(sorted_samples), num_points)

    # Interpolate the CDF
    cdf = np.interp(x, sorted_samples, ecdf_y)

    # Estimate the PDF by taking the numerical derivative of the CDF
    pdf = np.gradient(cdf, x)  # Use np.gradient for numerical differentiation

    return x, cdf, pdf