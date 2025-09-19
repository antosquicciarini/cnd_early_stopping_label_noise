import torch
import matplotlib.pyplot as plt


def differential_shannon_entropy_hist_approx(pmf, bin_width):
    """
    Computes the discrete Shannon entropy for each column of a PMF.

    Parameters:
    - pmf: Tensor of shape (num_samples, num_bins), representing the probability mass function.
    - bin_width: Tensor of shape (num_bins,), representing the width of each bin.

    Returns:
    - discrete_shannon_entropy: Tensor of shape (num_samples,), containing the entropy for each sample.
    """
    # Clamp PMF values to avoid log(0) issues
    pmf = pmf.clamp(min=1e-10)
    # Compute discrete Shannon entropy
    discrete_shannon_entropy = -torch.sum(pmf * torch.log2(pmf / bin_width.unsqueeze(1)), axis=1)

    return discrete_shannon_entropy

def KDE_entropy_approximation_OLD(samples):
    """
    Approximates the entropy of a set of samples using a Gaussian kernel with
    bandwidth estimated by Scott's Rule.

    Args:
        samples (Tensor): Shape (n_samples,).

    Returns:
        Tensor: Approximated entropy (scalar).
    """
    n = samples.shape[0]
    if n <= 1:
        # Entropy is zero or undefined for one or zero samples
        return torch.tensor(0.0, device=samples.device)

    # Estimate sigma using Scott's Rule for 1D data
    sigma = samples.std() * (n ** (-1 / 5))

    # Compute pairwise squared distances
    samples = samples.view(-1, 1)  # Shape (n_samples, 1)
    pairwise_distances = (samples - samples.t()) ** 2  # Shape (n_samples, n_samples)

    # Compute the kernel values
    kernel_values = torch.exp(-pairwise_distances / (2 * sigma ** 2))  # Shape (n_samples, n_samples)

    # Compute the kernel sum for each sample
    kernel_sum = kernel_values.sum(dim=1)  # Shape (n_samples,)

    # Avoid division by zero
    kernel_sum = kernel_sum.clamp(min=1e-10)

    # Compute the entropy approximation
    entropy_estimate = -torch.log(kernel_sum / n).mean()

    return entropy_estimate

def KDE_entropy_approximation(samples, bandwidth=None):
    """
    Approximates the entropy of a set of samples using a Gaussian kernel.
    
    Args:
        samples (Tensor): Shape (batch_size, n_neurons) for vectorized computation.
        bandwidth (Tensor, optional): Precomputed bandwidth for each neuron. Shape (n_neurons,).

    Returns:
        Tensor: Approximated entropy for each neuron (n_neurons,).
    """
    n_neurons, batch_size = samples.shape

    if batch_size <= 1:
        # Entropy is zero or undefined for one or zero samples
        return torch.zeros(n_neurons, device=samples.device)

    if bandwidth is None:
        bandwidth = samples.std(dim=1) * (batch_size ** (-1 / 5)) #CHECK THE SIGMA SIZE

    # Compute pairwise squared distances for all neurons in one go
    pairwise_distances = (samples[:, :, None] - samples[:, None, :]) ** 2  # Shape (batch_size, batch_size, n_neurons)
    kernel_values = torch.exp(-pairwise_distances / (2 * bandwidth ** 2).view(n_neurons, 1, 1))  # Shape (batch_size, batch_size, n_neurons)

    f_estimate = kernel_values.mean(dim=1)  # Shape (n_neurons, n_samples)
    f_estimate = f_estimate.clamp(min=1e-10)  # Avoid division by zero

    entropy_estimate = -torch.log(f_estimate).mean(dim=1)  # Averaging over samples
    
    return entropy_estimate  # Shape (n_neurons,) 


def KDE(samples, args, bandwidth=None, return_f_est=False):
    """
    Approximates the entropy of a set of samples using a Gaussian kernel.

    Args:
        samples (Tensor): Shape (batch_size, n_neurons) for vectorized computation.
        bandwidth (Tensor, optional): Precomputed bandwidth for each neuron. Shape (n_neurons,).
        return_density_estimator (bool): If True, returns a function to estimate the density on new points.

    Returns:
        Tensor: Approximated entropy for each neuron (n_neurons,).
        Function (optional): Density estimation function if `return_density_estimator` is True.
    """
    n_neurons, batch_size = samples.shape

    if bandwidth is None:
        bandwidth = samples.std(dim=1) * (batch_size ** (-1 / 5))  # Scott's rule for bandwidth

    if not return_f_est:
        return 0

    elif return_f_est:

        def density_estimator(new_points):
            """
            Estimates the density at new points using the precomputed bandwidth.

            Args:
                new_points (Tensor): Shape (batch_size_new, n_neurons).

            Returns:
                Tensor: Estimated density at new points. Shape (batch_size_new, n_neurons).
            """
            new_points = new_points.squeeze()

            # Compute pairwise distances between new_points and samples
            new_points_expanded = new_points[None, None, :]  # Shape: (batch_size_new, 1, n_neurons)
            samples_expanded = samples[:, :, None]  # Shape: (1, batch_size, n_neurons)
            pairwise_distances_new = (new_points_expanded - samples_expanded) ** 2  # Shape: (batch_size_new, batch_size, n_neurons)

            # Apply the Gaussian kernel
            kernel_values_new = torch.exp(-pairwise_distances_new / (2 * bandwidth ** 2).view(n_neurons, 1, 1))  # Shape: (batch_size_new, batch_size, n_neurons)

            # Compute the density estimate
            density_estimates = kernel_values_new.mean(dim=1)  # Shape: (batch_size_new, n_neurons)
            return density_estimates.clamp(min=1e-10)  # Avoid division by zero

        return density_estimator

    
def KDE_entropy_approximation_fft(samples, args, return_f_est=False, bandwidth=None):
    """
    Approximates the entropy of a set of samples using a Gaussian kernel with FFT.

    Args:
        samples (Tensor): Shape (n_neurons, batch_size).
        args: Configuration object with optional attributes like `CND_reg_type`.
        bandwidth (Tensor, optional): Bandwidth for each neuron. Shape (n_neurons,).

    Returns:
        Tensor: Approximated entropy for each neuron (n_neurons,).
    """
    if samples.ndim == 1:
        samples = samples.unsqueeze(0)  # Ensure 2D tensor

    n_neurons, batch_size = samples.shape
    device = samples.device

    if batch_size <= 1:
        return torch.zeros(n_neurons, device=device)  # Entropy is zero for one or zero samples

    # Estimate bandwidth if not provided
    if bandwidth is None:
        bandwidth = samples.std(dim=1) * (batch_size ** (-1 / 5))  # Scott's rule
    bandwidth = bandwidth.clamp(min=1e-5)  # Prevent bandwidth from being too small

    # Set frequency grid parameters
    if hasattr(args, "CND_reg_type") and "NoNorm" in args.CND_reg_type:
        f_max = 2**6
        s_max = 10.0 / bandwidth.min().item()
    else:
        f_max = 2**12
        s_max = 3.0 / bandwidth.min().item()  # Less than Nyquist frequency

    # Frequency grid and FFT scaling
    s = torch.linspace(-s_max, s_max, f_max, device=device)  # Frequency grid
    fft_scale = 1 / (2 * torch.pi * f_max)  # Scaling factor for FFT normalization

    # Define Gaussian kernel in the spatial domain and transform to frequency domain
    spatial_grid = torch.linspace(-4, 4, f_max, device=device)  # Amplitude grid
    kernel = (1 / (bandwidth.view(-1, 1) * (2 * torch.pi) ** 0.5)) * torch.exp(
        -0.5 * (spatial_grid / bandwidth.view(-1, 1)) ** 2
    )  # Gaussian kernel in spatial domain

    fft_kernel = torch.fft.fft(kernel, dim=1)  # Kernel in frequency domain

    # Compute FFT of samples
    padded_samples = torch.nn.functional.pad(samples, (0, f_max - batch_size))  # Pad to match FFT size
    fft_samples = torch.fft.fft(padded_samples, dim=1)  # FFT of padded samples

    # Perform convolution in frequency domain
    fft_convolution = fft_samples * fft_kernel  # Element-wise multiplication in frequency domain

    # Inverse FFT to return to spatial domain
    f_estimate = torch.fft.ifft(fft_convolution, dim=1).real[:, :batch_size]  # Truncate to valid range
    f_estimate = f_estimate.clamp(min=1e-10)  # Avoid log(0)

    # Compute entropy
    entropy_estimate = -torch.mean(torch.log(f_estimate), dim=1)  # Average entropy per neuron

    if return_f_est:
        return entropy_estimate, f_estimate
    else:
        return entropy_estimate  # Shape: (n_neurons,)


# a=0
# sorted_data, indices = torch.sort(torch.tensor(samples[a,:]))
# sorted_f_estimate = f_estimate[a,:].squeeze()[indices]
# plt.plot(sorted_data.cpu().detach(), sorted_f_estimate.cpu().detach()*10, label="KDE FFT differenciable", color="green", linewidth=2)
# #plt.plot(sorted_data.cpu().detach(), torch.log(sorted_f_estimate.cpu().detach()), label="KDE FFT differenciable", color="green", linewidth=2)


def KDE_entropy_approximation_incredible(samples, args, return_f_est=False, bandwidth=None):
    """
    Approximates the entropy of a set of samples using a Gaussian kernel with FFT.

    Args:
        samples (Tensor): Shape (n_neurons, batch_size).
        bandwidth (Tensor, optional): Bandwidth for each neuron. Shape (n_neurons,).

    Returns:
        Tensor: Approximated entropy for each neuron (n_neurons,).
    """
    if len(samples.shape) == 1:
        samples = samples.unsqueeze(0)
        
    n_neurons, batch_size = samples.shape
    device = samples.device

    if batch_size <= 1:
        # Entropy is zero or undefined for one or zero samples
        return torch.zeros(n_neurons, device=device)
    
    # Estimate bandwidth sigma if not provided
    if bandwidth is None:
        bandwidth = samples.std(dim=1) * (batch_size ** (-1 / 5))  # Scott's rule for bandwidth

    # Add an upper cap to avoid division by zero
    bandwidth = bandwidth.clamp(min=1e-5)  # Prevent bandwidth from being too small

    if hasattr(args, "CND_reg_type") and "NoNorm" in args.CND_reg_type:
        # Parameters for frequency grid
        f_max = 2**6
        s_max = 10.0/bandwidth.min().item()
    else:
        # Parameters for frequency grid
        f_max = 2**6
        s_max = 3/bandwidth.min().item() #less than Nyquist freq batch_size/2
    
    s = torch.linspace(-s_max, s_max, f_max, device=device)  # Frequency grid

    # Compute the Empirical Characteristic Function (ECF)
    s_expanded = s[None, None, :]  # Shape: (1, 1, num_freq)
    samples_expanded = samples[:, :, None]  # Shape: (n_neurons, batch_size, 1)
    exponentials = torch.exp(-1j * s_expanded * samples_expanded)  # Shape: (n_neurons, batch_size, num_freq)
    ecf = torch.sum(exponentials, dim=1)  # Sum over batch_size, Shape: (n_neurons, num_freq)

    # Compute the Fourier Transform of the kernel (Gaussian)
    h = bandwidth.view(n_neurons, 1)  # Shape: (n_neurons, 1)
    kernel_ft = torch.exp(-0.5 * (h * s[None, :]) ** 2)  # Shape: (n_neurons, num_freq)
    if hasattr(args, "CND_reg_type") and "hKern" in args.CND_reg_type:
        kernel_ft = h*kernel_ft  # Shape: (n_neurons, num_freq)

    # Compute the product in the frequency domain
    f_hat = ecf * kernel_ft  # Shape: (n_neurons, num_freq)

    # Compute f(x_j) at the sample points x_j
    # Compute exponentials for inverse Fourier transform
    exponentials_ifft = torch.exp(1j * s_expanded * samples_expanded)  # Shape: (n_neurons, batch_size, num_freq)

    # Multiply f_hat with exponentials and sum over frequencies
    f_hat_expanded = f_hat[:, None, :]  # Shape: (n_neurons, 1, num_freq)
    product = f_hat_expanded * exponentials_ifft  # Shape: (n_neurons, batch_size, num_freq)
    f_estimate = (1 / (2 * torch.pi * batch_size)) * torch.sum(product, dim=2).real  # Shape: (n_neurons, batch_size)

    # #Compute entropy per neuron
    # if (f_estimate < 1e-10).any():
    #     print(f"Values below 1e-10 found: {f_estimate[f_estimate < 1e-10]}")

    f_estimate = f_estimate.clamp(min=1e-10)  # Avoid log(0)
    entropy_estimate = -torch.mean(torch.log(f_estimate), dim=1)  # Shape: (n_neurons,)

    if return_f_est:
        return entropy_estimate, f_estimate
    else:
        return entropy_estimate  # Shape: (n_neurons,)
    
def KDE_entropy_approximation_incredible_2(samples, bandwidth=None):
    """
    Approximates the entropy of a set of samples using a Gaussian kernel with FFT.

    Args:
        samples (Tensor): Shape (n_neurons, batch_size).
        bandwidth (Tensor, optional): Bandwidth for each neuron. Shape (n_neurons,).

    Returns:
        Tensor: Approximated entropy for each neuron (n_neurons,).
    """
    n_neurons, batch_size = samples.shape

    if batch_size <= 1:
        # Entropy is zero or undefined for one or zero samples
        return torch.zeros(n_neurons, device=samples.device)
    
    # Standardise the data points
    #samples = (samples - samples.mean(dim=1, keepdim=True)) / bandwidth.view(-1, 1)
    # samples = (samples - samples.mean(dim=1, keepdim=True)) / samples.std(dim=1, keepdim=True).view(-1, 1)

    # Estimate bandwidth sigma if not provided
    if bandwidth is None:
        bandwidth = 10.0#samples.std(dim=1) * (batch_size ** (-1 / 5))  # Scott's rule for bandwidth

    # Define grid for FFT
    grid_size = 2**6
    grid = torch.linspace(0, 1, grid_size, device=samples.device)  # Create a base grid
    grid = grid.unsqueeze(0).repeat(n_neurons, 1)  # Shape: [N, n_grid_points]
    grid = -4 * bandwidth.unsqueeze(1) + grid * 8 * bandwidth.unsqueeze(1)

    # Compute Gaussian kernel on the grid
    kernel = (1/(bandwidth.view(-1, 1)*(torch.pi*2)**(1/2))) *torch.exp(-0.5 * (grid / bandwidth.view(-1, 1)) ** 2)  # Shape (n_neurons, grid_size)
    kernel = kernel / kernel.sum(dim=1, keepdim=True)  # Normalize kernel

    # Compute FFT of samples and kernel
    fft_samples = torch.fft.fft(samples, n=grid_size, dim=1)  # Shape (n_neurons, grid_size)
    fft_kernel = torch.fft.fft(kernel, dim=1)  # Shape (n_neurons, grid_size)

    # Convolution in frequency domain
    f_estimate = torch.fft.ifft(fft_samples * fft_kernel, dim=1).real  # Back to spatial domain
    f_estimate = f_estimate[:, :batch_size]  # Take valid part of convolution (original sample size)

    # Compute KDE and entropy per neuron
    #kde = convolved.mean(dim=1)  # Average over batch samples for each neuron
    f_estimate = f_estimate.clamp(min=1e-10)  # Avoid log(0)
    entropy_estimate = (-torch.log(f_estimate).mean(dim=1))  # Entropy per neuron

    return entropy_estimate  # Shape (n_neurons,)

def KDE_entropy_approximation_fourier_transform(samples, args, return_f_est=False, bandwidth=None):

    if len(samples.shape) == 1:
        samples = samples.unsqueeze(0)
        
    n_neurons, batch_size = samples.shape
    device = samples.device

    if batch_size <= 1:
        # Entropy is zero or undefined for one or zero samples
        return torch.zeros(n_neurons, device=device)
    
    # Estimate bandwidth sigma if not provided
    if bandwidth is None:
        bandwidth = samples.std(dim=1) * (batch_size ** (-1 / 5))  # Scott's rule for bandwidth

    # Add an upper cap to avoid division by zero
    bandwidth = bandwidth.clamp(min=1e-5)  # Prevent bandwidth from being too small

    # Parameters for frequency grid

    if hasattr(args, "CND_reg_type") and "NoNorm" in args.CND_reg_type:
        f_max = 2**6
        s_max = 10.0 / bandwidth.min().item()
        s = torch.linspace(-s_max, s_max, f_max, device=device)   # Shape: (n_neurons, f_max)
        s = s[None, :]
    else:
        f_max = 2**6  # Number of frequency points
        s_max = 10.0 / bandwidth  # Compute s_max individually for each bandwidth
        s = torch.stack([torch.linspace(-s_max_i.item(), s_max_i.item(), f_max, device=device) for s_max_i in s_max], dim=0)


    # Compute the Empirical Characteristic Function (ECF)
    s_expanded = s.unsqueeze(1)  # Shape: (n_neurons, 1, f_max)
    samples_expanded = samples[:, :, None]  # Shape: (n_neurons, batch_size, 1)
    exponentials = torch.exp(-1j * s_expanded * samples_expanded)  # Shape: (n_neurons, batch_size, num_freq)
    ecf = torch.sum(exponentials, dim=1)  # Sum over batch_size, Shape: (n_neurons, num_freq)

    # Compute the Fourier Transform of the kernel (Gaussian)
    h = bandwidth.view(n_neurons, 1)  # Shape: (n_neurons, 1)
    kernel_ft = torch.exp(-0.5 * (h * s) ** 2)  # Shape: (n_neurons, num_freq)
    if hasattr(args, "CND_reg_type") and "hKern" in args.CND_reg_type:
        kernel_ft = h*kernel_ft  # Shape: (n_neurons, num_freq)

    # Compute the product in the frequency domain
    f_hat = ecf * kernel_ft  # Shape: (n_neurons, num_freq)

    if not return_f_est:

        exponentials_ifft = torch.exp(-1j * s_expanded * samples_expanded)  # Shape: (n_neurons, batch_size, num_freq)

        f_hat_expanded = f_hat[:, None, :]  # Shape: (n_neurons, 1, num_freq)
        product = f_hat_expanded * exponentials_ifft  # Shape: (n_neurons, batch_size, num_freq)
        f_estimate = (1 / (2 * torch.pi * batch_size)) * torch.sum(product, dim=2).real  # Shape: (n_neurons, batch_size)

        f_estimate = torch.where(f_estimate < 1e-7, torch.tensor(1.0, device=f_estimate.device), f_estimate)  # Replace values under 1e-7 with 1.0
        f_estimate = f_estimate.clamp(min=1e-10)  # Avoid log(0)
        entropy_estimate = -torch.mean(torch.log(f_estimate), dim=1)

        return entropy_estimate  # Shape: (n_neurons,)
    
    elif return_f_est:

        # Function to compute density estimates
        def density_estimate(points):
            """
            Computes the density estimate at given points.

            Args:
                points (Tensor): Shape (n_neurons, n_points), the points where density is estimated.

            Returns:
                Tensor: Estimated densities at the specified points. Shape: (n_neurons, n_points).
            """
            points_expanded = points[:, :, None]  # Shape: (n_neurons, n_points, 1)
            exponentials_ifft = torch.exp(1j * s_expanded * points_expanded)  # Shape: (n_neurons, n_points, num_freq)
            f_hat_expanded = f_hat[:, None, :]  # Shape: (n_neurons, 1, num_freq)
            product = f_hat_expanded * exponentials_ifft  # Shape: (n_neurons, n_points, num_freq)
            f_estimate = (1 / (2 * torch.pi * points.shape[1])) * torch.sum(product, dim=2).real  # Shape: (n_neurons, n_points)
            return f_estimate.clamp(min=1e-10)  # Avoid log(0)

        return density_estimate




def KDE_entropy_approximation_fourier_transform_old(samples, args, return_f_est=False, bandwidth=None):
    """
    Approximates the entropy of a set of samples using a Gaussian kernel with FFT.

    Args:
        samples (Tensor): Shape (n_neurons, batch_size).
        bandwidth (Tensor, optional): Bandwidth for each neuron. Shape (n_neurons,).

    Returns:
        Tensor: Approximated entropy for each neuron (n_neurons,).
    """
    if len(samples.shape) == 1:
        samples = samples.unsqueeze(0)
        
    n_neurons, batch_size = samples.shape
    device = samples.device

    if batch_size <= 1:
        # Entropy is zero or undefined for one or zero samples
        return torch.zeros(n_neurons, device=device)
    
    # Estimate bandwidth sigma if not provided
    if bandwidth is None:
        bandwidth = samples.std(dim=1) * (batch_size ** (-1 / 5))  # Scott's rule for bandwidth

    # Add an upper cap to avoid division by zero
    bandwidth = bandwidth.clamp(min=1e-5)  # Prevent bandwidth from being too small


    # Parameters for frequency grid
    f_max = 2**6
    s_max = 10.0 / bandwidth.min().item() if hasattr(args, "CND_reg_type") and "NoNorm" in args.CND_reg_type else 3.0 / bandwidth.min().item()    
    s = torch.linspace(-s_max, s_max, f_max, device=device)  # Frequency grid

    # Compute the Empirical Characteristic Function (ECF)
    s_expanded = s[None, None, :]  # Shape: (1, 1, num_freq)
    samples_expanded = samples[:, :, None]  # Shape: (n_neurons, batch_size, 1)
    exponentials = torch.exp(-1j * s_expanded * samples_expanded)  # Shape: (n_neurons, batch_size, num_freq)
    ecf = torch.sum(exponentials, dim=1)  # Sum over batch_size, Shape: (n_neurons, num_freq)

    # Compute the Fourier Transform of the kernel (Gaussian)
    h = bandwidth.view(n_neurons, 1)  # Shape: (n_neurons, 1)
    kernel_ft = torch.exp(-0.5 * (h * s[None, :]) ** 2)  # Shape: (n_neurons, num_freq)
    if hasattr(args, "CND_reg_type") and "hKern" in args.CND_reg_type:
        kernel_ft = h*kernel_ft  # Shape: (n_neurons, num_freq)

    # Compute the product in the frequency domain
    f_hat = ecf * kernel_ft  # Shape: (n_neurons, num_freq)

    # Compute f(x_j) at the sample points x_j
    # Compute exponentials for inverse Fourier transform
    exponentials_ifft = torch.exp(-1j * s_expanded * samples_expanded)  # Shape: (n_neurons, batch_size, num_freq)

    # Multiply f_hat with exponentials and sum over frequencies
    f_hat_expanded = f_hat[:, None, :]  # Shape: (n_neurons, 1, num_freq)
    product = f_hat_expanded * exponentials_ifft  # Shape: (n_neurons, batch_size, num_freq)
    f_estimate = (1 / (2 * torch.pi * batch_size)) * torch.sum(product, dim=2).real  # Shape: (n_neurons, batch_size)

    # #Compute entropy per neuron
    # if (f_estimate < 1e-10).any():
    #     print(f"Values below 1e-10 found: {f_estimate[f_estimate < 1e-10]}")

    f_estimate = f_estimate.clamp(min=1e-10)  # Avoid log(0)
    entropy_estimate = -torch.mean(torch.log(f_estimate), dim=1)  # Shape: (n_neurons,)

    # Function to compute density estimates
    def density_estimate(points):
        """
        Computes the density estimate at given points.

        Args:
            points (Tensor): Shape (n_neurons, n_points), the points where density is estimated.

        Returns:
            Tensor: Estimated densities at the specified points. Shape: (n_neurons, n_points).
        """
        points_expanded = points[:, :, None]  # Shape: (n_neurons, n_points, 1)
        exponentials_ifft = torch.exp(1j * s_expanded * points_expanded)  # Shape: (n_neurons, n_points, num_freq)
        f_hat_expanded = f_hat[:, None, :]  # Shape: (n_neurons, 1, num_freq)
        product = f_hat_expanded * exponentials_ifft  # Shape: (n_neurons, n_points, num_freq)
        f_estimate = (1 / (2 * torch.pi * points.shape[1])) * torch.sum(product, dim=2).real  # Shape: (n_neurons, n_points)
        return f_estimate.clamp(min=1e-10)  # Avoid log(0)


    if return_f_est:
        return entropy_estimate, density_estimate
    else:
        return entropy_estimate  # Shape: (n_neurons,)


# a=2
# sorted_data, indices = torch.sort(torch.tensor(samples[a,:]))
# sorted_f_estimate = f_estimate[a,:].squeeze()[indices]
# plt.plot(sorted_data.cpu().detach(), sorted_f_estimate.cpu().detach()*10, label="KDE FFT differenciable", color="green", linewidth=2)
# plt.plot(sorted_data.cpu().detach(), torch.log(sorted_f_estimate.cpu().detach()), label="KDE FFT differenciable", color="green", linewidth=2)

# plt.legend()
# plt.grid()

def KDE_entropy_approximation_fourier_transform_on_grid(samples, return_f_est=False, bandwidth=None):
    """
    Approximates the density of a set of samples at specific grid points using a Gaussian kernel with FFT.

    Args:
        samples (Tensor): Shape (n_neurons, batch_size), the original samples.
        grid_points (Tensor): Shape (n_neurons, n_grid_points), points where the density is estimated.
        bandwidth (Tensor, optional): Bandwidth for each neuron. Shape (n_neurons,).

    Returns:
        Tensor: Estimated densities at grid points. Shape (n_neurons, n_grid_points).
    """
    if len(samples.shape) == 1:
        samples = samples.unsqueeze(0)
        
    n_neurons, batch_size = samples.shape

    device = samples.device

    # Estimate bandwidth sigma if not provided
    if bandwidth is None:
        bandwidth = samples.std(dim=1) * (batch_size ** (-1 / 5))  # Scott's rule for bandwidth

    bandwidth = bandwidth.clamp(min=1e-5)  # Prevent small bandwidth values

    # Define grid points
    n_grid_points = 2**8  # Set the number of grid points
    samples_min, _ = samples.min(dim=1) 
    samples_max, _ = samples.max(dim=1)  
    samples_min = samples_min - 4*bandwidth
    samples_max = samples_min + 4*bandwidth
    grid_points = torch.linspace(0, 1, n_grid_points, device=device)  # Base grid shape: (n_grid_points,)
    grid_points = grid_points.unsqueeze(0).repeat(samples_min.shape[0], 1)  # Shape: [N, n_grid_points]
    grid_points = samples_min.unsqueeze(1) + grid_points * (samples_max.unsqueeze(1) - samples_min.unsqueeze(1))  # Shape: (N, n_grid_points)

    # Parameters for frequency grid
    f_max = 2**6
    s_max = 10.0 / bandwidth.min().item()
    s = torch.linspace(-s_max, s_max, f_max, device=device)  # Frequency grid

    # Compute the Empirical Characteristic Function (ECF)
    s_expanded = s[None, None, :]  # Shape: (1, 1, num_freq)
    samples_expanded = samples[:, :, None]  # Shape: (n_neurons, batch_size, 1)
    exponentials = torch.exp(-1j * s_expanded * samples_expanded)  # Shape: (n_neurons, batch_size, num_freq)
    ecf = torch.sum(exponentials, dim=1)  # Sum over batch_size, Shape: (n_neurons, num_freq)

    # Compute the Fourier Transform of the kernel (Gaussian)
    h = bandwidth.view(n_neurons, 1)  # Shape: (n_neurons, 1)
    kernel_ft = torch.exp(-0.5 * (h * s[None, :]) ** 2)  # Shape: (n_neurons, num_freq)

    # Compute the product in the frequency domain
    f_hat = ecf * kernel_ft  # Shape: (n_neurons, num_freq)

    # Compute f(x) at the grid points
    grid_points_expanded = grid_points[:, :, None]  # Shape: (n_neurons, n_grid_points, 1)
    exponentials_ifft = torch.exp(1j * s_expanded * grid_points_expanded)  # Shape: (n_neurons, n_grid_points, num_freq)

    f_hat_expanded = f_hat[:, None, :]  # Shape: (n_neurons, 1, num_freq)
    product = f_hat_expanded * exponentials_ifft  # Shape: (n_neurons, n_grid_points, num_freq)
    f_estimate = (1 / (2 * torch.pi * n_grid_points)) * torch.sum(product, dim=2).real  # Shape: (n_neurons, n_grid_points)

    # Compute entropy per neuron
    f_estimate = f_estimate.clamp(min=1e-10)  # Avoid log(0)
    entropy_estimate = -torch.mean(torch.log(f_estimate), dim=1)  # Shape: (n_neurons,)

    if return_f_est:
        return entropy_estimate, f_estimate
    else:
        return entropy_estimate  # Shape: (n_neurons,)

def GaussianLike_entropy_approximation(samples):
    """
    Approximates the entropy of a set of samples using Gaussian analytic equation.
    
    Args:
        samples (Tensor): Shape (n_neurons, n_samples) for vectorized computation.

    Returns:
        Tensor: Approximated entropy for each neuron (n_neurons,).
    """
    # Compute the standard deviation along dimension 1
    sigmas = torch.std(samples, dim=1)

    # Avoid log(0) or other numerical issues
    sigmas = torch.clamp(sigmas, min=1e-10)

    # Gaussian Shannon entropy formula
    return 0.5 * torch.log(2 * torch.pi * torch.e * sigmas**2)





def KDE_entropy_approximation_fourier_old(samples, bandwidth=None):
    """
    Approximates the entropy of a set of samples using a Gaussian kernel with FFT.

    Args:
        samples (Tensor): Shape (n_neurons, batch_size).
        bandwidth (Tensor, optional): Bandwidth for each neuron. Shape (n_neurons,).

    Returns:
        Tensor: Approximated entropy for each neuron (n_neurons,).
    """
    n_neurons, batch_size = samples.shape
    device = samples.device

    if batch_size <= 1:
        # Entropy is zero or undefined for one or zero samples
        return torch.zeros(n_neurons, device=device)
    
    # Estimate bandwidth sigma if not provided
    if bandwidth is None:
        bandwidth = samples.std(dim=1) * (batch_size ** (-1 / 5))  # Scott's rule for bandwidth
    
    # Add an upper cap to avoid division by zero
    bandwidth = bandwidth.clamp(min=1e-5)  # Prevent bandwidth from being too small

    # Parameters for frequency grid
    f_max = 2**6
    s_max = 10.0 / bandwidth.min().item()  # Adjust s_max based on bandwidth
    s = torch.linspace(-s_max, s_max, f_max, device=device)  # Frequency grid

    # Compute the Empirical Characteristic Function (ECF)
    s_expanded = s[None, None, :]  # Shape: (1, 1, num_freq)
    samples_expanded = samples[:, :, None]  # Shape: (n_neurons, batch_size, 1)
    exponentials = torch.exp(-1j * s_expanded * samples_expanded)  # Shape: (n_neurons, batch_size, num_freq)
    ecf = torch.sum(exponentials, dim=1)  # Sum over batch_size, Shape: (n_neurons, num_freq)

    # Compute the Fourier Transform of the kernel (Gaussian)
    h = bandwidth.view(n_neurons, 1)  # Shape: (n_neurons, 1)
    kernel_ft = h * torch.exp(-0.5 * (h * s[None, :]) ** 2)  # Shape: (n_neurons, num_freq)

    # Compute the product in the frequency domain
    f_hat = ecf * kernel_ft  # Shape: (n_neurons, num_freq)

    # Compute f(x_j) at the sample points x_j
    # Compute exponentials for inverse Fourier transform
    exponentials_ifft = torch.exp(1j * s_expanded * samples_expanded)  # Shape: (n_neurons, batch_size, num_freq)

    # Multiply f_hat with exponentials and sum over frequencies
    f_hat_expanded = f_hat[:, None, :]  # Shape: (n_neurons, 1, num_freq)
    product = f_hat_expanded * exponentials_ifft  # Shape: (n_neurons, batch_size, num_freq)
    f_estimate = (1 / (2 * torch.pi * batch_size)) * torch.sum(product, dim=2).real  # Shape: (n_neurons, batch_size)

    # Compute entropy per neuron
    f_estimate = f_estimate.clamp(min=1e-10)  # Avoid log(0)
    entropy_estimate = -torch.mean(torch.log(f_estimate), dim=1)  # Shape: (n_neurons,)

    return entropy_estimate  # Shape: (n_neurons,)