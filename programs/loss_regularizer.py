import torch
from SE_approximatiors import KDE_entropy_approximation
from jsd import jsd_KDE, jsd_GaussianLike, jsd_trapezoidal_kl

def apply_regularizers(loss, model, labels, pre_activations, epoch_per, args):

    def l1_regularization(model, lambda_l1):
        l1_penalty = 0
        for param in model.parameters():
            l1_penalty += torch.sum(torch.abs(param))
        return lambda_l1 * l1_penalty

    if hasattr(args, "l1_regularization"):
        loss = loss + l1_regularization(model, args.l1_regularization)
    
    elif hasattr(args, "CND_reg_type") and "KDE_FFT" in args.CND_reg_type and args.CND_reg_gamma!=0 and args.batch_size == pre_activations.shape[0]:
        #loss = loss - args.jsd_regularizer * jsd_regularizer(pre_activations, labels, args)
        if hasattr(args, "CND_reg_type") and "linear_decr" in args.CND_reg_type:
            loss = loss - jsd_KDE(pre_activations, labels, args, return_mean=True, model_eval=False) #jsd_KDE(pre_activations, labels, args, return_mean=True)
        elif hasattr(args, "CND_reg_type") and "incredible" in args.CND_reg_type:
            jsd_value = jsd_KDE(pre_activations, labels, args, return_mean=True, model_eval=False)
            loss = loss - args.CND_reg_gamma * jsd_value #jsd_KDE(pre_activations, labels, args, return_mean=True)
        else:
            jsd_value = jsd_trapezoidal_kl(pre_activations, labels, args, return_mean=True, model_eval=False)
            loss = loss - args.CND_reg_gamma * jsd_value #jsd_KDE(pre_activations, labels, args, return_mean=True)

    elif hasattr(args, "CND_reg_type") and args.CND_reg_type == "GL" and args.CND_reg_gamma!=0  and args.batch_size == pre_activations.shape[0]:
        loss = loss - args.CND_reg_gamma * jsd_GaussianLike(pre_activations, labels, args, return_mean=True)

    elif hasattr(args, "CND_reg_type") and args.CND_reg_type == "mean" and args.CND_reg_gamma!=0  and args.batch_size == pre_activations.shape[0]:
        loss = loss - args.CND_reg_gamma * CND_reg_mean(pre_activations, labels, args)

    elif hasattr(args, "CND_reg_type") and args.CND_reg_type == "mean_All_VS_All" and args.CND_reg_gamma!=0  and args.batch_size == pre_activations.shape[0]:
        loss = loss - args.CND_reg_gamma * CND_reg_mean_All_VS_All(pre_activations, labels, args)

    return loss


def jsd_regularizer_OLD(pre_activations, outputs, args):
    """
    Computes the regularizer that maximizes the JSD between class-conditional distributions
    of pre-activation outputs for each neuron.

    Args:
        pre_activations (Tensor): Pre-activation outputs from a layer. Shape (batch_size, n_neurons).
        labels (Tensor): Class labels for each sample. Shape (batch_size,).
        num_classes (int): Number of classes.
        sigma (float): Bandwidth parameter for the entropy approximation.

    Returns:
        Tensor: The regularizer scalar value.
    """
    _, labels = torch.max(outputs, 1)
    batch_size, n_neurons = pre_activations.shape
    device = pre_activations.device

    # Initialize the regularizer
    regularizer = torch.tensor(0.0, device=device)

    # Loop over each neuron
    for neuron_idx in range(n_neurons):
        # Get the pre-activation outputs for this neuron
        neuron_outputs = pre_activations[:, neuron_idx]  # Shape (batch_size,)

        # Compute the entropy of the mixture distribution
        H_M = KDE_entropy_approximation(neuron_outputs)

        # Compute the entropy of each class-conditional distribution
        H_c_list = []
        for class_idx in range(args.num_classes):
            # Get the samples belonging to the current class
            class_mask = (labels == class_idx)
            class_samples = neuron_outputs[class_mask]
            if class_samples.numel() == 0:
                continue  # Skip if no samples for this class
            H_c = KDE_entropy_approximation(class_samples)
            H_c_list.append(H_c)

        if len(H_c_list) == 0:
            continue  # Skip if no classes have samples

        # Compute the average entropy over classes
        H_C = torch.stack(H_c_list).mean()

        # Compute the JSD for this neuron
        JSD = H_M - H_C

        # Accumulate the regularizer (maximize JSD)
        regularizer += JSD

    # Normalize the regularizer by the number of neurons
    regularizer = regularizer / n_neurons

    return regularizer

def jsd_regularizer(pre_activations, labels, args):
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
    H_M = KDE_entropy_approximation(pre_activations)

    # Initialize a sum for the entropy of each class-conditional distribution
    H_C_sum = torch.zeros(n_neurons, device=device)
    num_valid_classes = 0

    # Loop over each class to compute the class-conditional entropies
    for class_idx in range(args.num_classes):
        valid_samples = pre_activations[:, labels == class_idx]
        
        if valid_samples.numel() == 0:
            continue  # Skip if no samples for this class

        # Sum the entropies of valid classes
        H_C_sum += KDE_entropy_approximation(valid_samples)
        num_valid_classes += 1

    if num_valid_classes == 0:
        return torch.tensor(0.0, device=device)

    # Calculate average entropy over classes
    H_C = H_C_sum / num_valid_classes  # Shape (n_neurons,)

    # Compute the JSD across neurons and then average
    JSD = H_M - H_C
    regularizer = JSD.mean()  # Scalar

    return regularizer


def CND_reg_mean(pre_activations, labels, args):
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
    device = pre_activations.device

    # Compute the global mean of pre-activations
    global_mean = pre_activations.mean(dim=0)  # Shape (n_neurons,)

    # Initialize a tensor for class-conditional means
    class_means = torch.zeros((args.num_classes, n_neurons), device=device)

    # Compute the mean activation for each class
    for class_idx in range(args.num_classes):
        class_mask = labels == class_idx  # Shape (batch_size,)
        class_samples = pre_activations[class_mask]  # Shape (num_samples_for_class, n_neurons)
        if class_samples.numel() > 0:
            class_means[class_idx] = class_samples.mean(dim=0)  # Mean along the batch dimension

    # Compute the absolute differences between the global mean and class means
    abs_diffs = torch.abs(class_means - global_mean)  # Shape (num_classes, n_neurons)

    # Compute the mean over classes and neurons
    regularizer = abs_diffs.mean()  # Scalar

    return regularizer


def CND_reg_mean_All_VS_All(pre_activations, labels, args):
    """
    Computes the JSD regularizer across all neurons in a vectorized manner.

    Args:
        pre_activations (Tensor): Shape (batch_size, n_neurons).
        labels (Tensor): Class labels for each sample. Shape (batch_size,).
        args: Argument object with num_classes attribute.

    Returns:
        Tensor: Regularizer scalar value.
    """
    batch_size, n_neurons = pre_activations.shape
    device = pre_activations.device

    # Transpose pre_activations for easier processing
    pre_activations = pre_activations.T  # Shape (n_neurons, batch_size)

    # Initialize a tensor to hold the mean activations for each class
    class_means = torch.zeros((args.num_classes, n_neurons), device=device)

    # Compute the mean activation for each class
    for class_idx in range(args.num_classes):
        class_mask = (labels == class_idx)  # Shape (batch_size,)
        class_samples = pre_activations[:, class_mask]  # Shape (n_neurons, num_samples_for_class)
        if class_samples.numel() > 0:
            class_means[class_idx] = class_samples.mean(dim=1)

    # Compute pairwise absolute differences between class means
    diffs = class_means.unsqueeze(0) - class_means.unsqueeze(1)  # Shape (num_classes, num_classes, n_neurons)
    abs_diffs = torch.abs(diffs).mean(dim=2)  # Mean over neurons (axis 2)

    # Average over all class pairs
    regularizer = abs_diffs.mean()  # Scalar

    return regularizer

