import torch
import matplotlib.pyplot as plt

from jsd import (
    jsd_bins_hist, 
    jsd_bins_hist_adaptive,
    mean_approx_JSD, 
    multi_distr_jsd_kNN, 
    topK_jsd_bins_hist_approx, 
    jsd_GaussianLike, 
    jsd_KDE,
    jsd_trapezoidal_kl,
    jsd_trapezoidal_kl_adaptive,
    jsd_trapezoidal_kl_adaptive_chebyshev
)
import logging

def cnd(loader, model, device, performances_dict, metric, args, logger):
    """
    Compute Conditional Neuron Divergence (CND) metrics for a given model and dataset loader.

    Parameters:
    - loader: DataLoader for the dataset.
    - model: PyTorch model for which CND is computed.
    - device: Device (e.g., 'cuda' or 'cpu') for computation.
    - performances_dict: Dictionary to store performance metrics.
    - args: Arguments containing configurations.
    - logger: Logger instance for reporting.

    Returns:
    - performances_dict: Updated dictionary with computed CND metrics.
    """

    model.eval()
    with torch.no_grad():
        # Compute neuron pre-activations and associated data
        pre_activations, labels, predictions = calculate_neuron_pre_activations(loader, model, device, args)

        # filter only well classified samples
        if args.num_classes>10 or getattr(args, "exclude_CND_correct_filter", False): #Second condition for dataset with many classes
            mask = torch.ones_like(labels, dtype=torch.bool)  # All True values
        else:
            mask = labels == predictions

        pre_activations = pre_activations[mask]
        labels = labels[mask]

        # Compute CND metrics
        if not isinstance(args.CND_type, list):
            args.CND_type = [args.CND_type]

        for cnd_type in args.CND_type:

            cnd_result = compute_cnd_metrics(cnd_type, pre_activations, labels, metric, args)
            performances_dict[metric+"_"+cnd_type] = cnd_result
            logger.info(f"{cnd_type} -- Mean {metric}: {torch.mean(cnd_result):.4f}")

        # if args.current_epoch // 10 ==0:
        #     generate_PDF_plots(pre_activations, metric, args)

    return performances_dict, predictions


def calculate_neuron_pre_activations(loader, model, device, args, max_images=10000):
    """
    Calculate neuron pre-activations, labels, and predictions for a given loader.

    Parameters:
    - loader: DataLoader for the dataset.
    - model: PyTorch model for which pre-activations are calculated.
    - device: Device (e.g., 'cuda' or 'cpu') for computation.
    - args: Arguments containing configurations.
    - max_images: Maximum number of images to process. Defaults to 10,000.

    Returns:
    - pre_activations: Tensor containing pre-activations for all samples.
    - labels: Tensor containing labels for all samples.
    - predictions: Tensor containing predictions for all samples.
    """

    args.layer_indexes = getattr(args, "layer_indexes", [])
    args.neuron_indexes = getattr(args, "neuron_indexes", [])

    # Handle layer and neuron indexes
    layer_indexes = slice(None) if not args.layer_indexes else args.layer_indexes

    # Initialize variables
    pre_activation_list, label_list, prediction_list = [], [], []
    processed_images = 0

    for images, labels, idx in loader:
        # Stop if the maximum number of images is reached
        if processed_images >= max_images:
            break

        # Move data to device
        images, labels = images.to(device), labels.to(device)

        # Forward pass to get activations and predictions
        logit, pre_activation = model(
            images, idx,
            return_intermediates=True,
            layer_indexes=layer_indexes,
            por_neuron = getattr(args, "por_neurons_x_layer_preact", 1.),
            CND_reg_only_last_layer = getattr(args, "CND_reg_only_last_layer", False)
        )
        _, predictions = torch.max(logit, dim=1)

        # Store activations, labels, and predictions
        pre_activation_list.append(pre_activation.cpu())
        label_list.append(labels.cpu())
        prediction_list.append(predictions.cpu())

        # Update the count of processed images
        processed_images += images.size(0)

    # Combine results from all batches
    pre_activations = torch.cat(pre_activation_list, dim=0)
    labels = torch.cat(label_list)
    predictions = torch.cat(prediction_list)

    return pre_activations, labels, predictions


def update_neuron_activations(pre_activation, labels, predictions, neuron_activations_dict, 
                              max_activation, min_activation, args):
    """
    Update neuron activation statistics based on model predictions and labels.
    """
    for class_idx in range(args.num_classes):
        mask = (labels == class_idx) & (predictions == labels) if getattr(args,'CND_well_classified_filter', True) else (labels == class_idx)
        if mask.sum() == 0:
            continue
        
        neuron_activations = pre_activation[mask].detach().cpu()

        # Update max and min activations
        max_values = torch.max(neuron_activations, dim=0).values
        min_values = torch.min(neuron_activations, dim=0).values
        max_activation = torch.maximum(max_activation, max_values)
        min_activation = torch.minimum(min_activation, min_values)

        # Update or initialize activations per class
        if class_idx in neuron_activations_dict:
            neuron_activations_dict[class_idx] = torch.cat(
                [neuron_activations_dict[class_idx], neuron_activations], dim=0
            )
        else:
            neuron_activations_dict[class_idx] = neuron_activations


def compute_cnd_metrics(cnd_type, pre_activations, labels, metric, args):
    """
    Compute various Conditional Neuron Divergence (CND) metrics.
    """

    if cnd_type == "mean":
        cnd_result = mean_approx_JSD(pre_activations, labels, args)

    elif cnd_type == "PMF":
        cnd_result = jsd_bins_hist(pre_activations, labels, args)

    elif cnd_type == "kNN":
        cnd_result = multi_distr_jsd_kNN(pre_activations, labels, args)

    elif cnd_type == "KDE":
        #cnd_result = jsd_trapezoidal_kl_adaptive(pre_activations, labels, metric, args,  model_eval=True)
        cnd_result = jsd_trapezoidal_kl_adaptive_chebyshev(pre_activations, labels, metric, args,  model_eval=True)

    elif cnd_type == "GL":
        cnd_result = jsd_GaussianLike(pre_activations, labels, args)

    elif "top" in cnd_type:
        cnd_result = topK_jsd_bins_hist_approx(pre_activations, labels, args)

    return cnd_result






# import torch
# from jsd import multi_distr_jsd, mean_approx_JSD, jsd_bins_hist, multi_distr_jsd_kNN, topK_jsd_bins_hist_approx, jsd_GaussianLike, jsd_KDE
# import matplotlib.pyplot as plt
# import logging

# def cnd(loader, model, device, performances_dict, args, logger):

#     model.eval()
#     with torch.no_grad():

#         if len(args.layer_indexes) == 0: #Save All layers
#             layer_indexes = slice(None)
#         else:
#             layer_indexes = args.layer_indexes

#         if len(args.neuron_indexes) == 0: #Save All neurons activations X each layer
#             neurons_indexes = slice(None)
#         else:
#             neurons_indexes = args.neuron_indexes

#         neuron_activations_dict = {} # if hasattr(neuron_activations_dict, ii): neuron_activations_dict[ii] = torch.cat(neuron_activations_dict[ii], neuron_activations)
#         if isinstance(layer_indexes, slice):
#             # If layer_indexes is a slice(None), select all neurons
#             n_neurons = sum(args.neurs_x_hid_lyr.values())
#         else:
#             # Otherwise, sum only the specified layers
#             n_neurons = sum([value for key, value in args.neurs_x_hid_lyr.items() if key in layer_indexes])

#         max_activation = torch.zeros(n_neurons)    
#         min_activation = torch.zeros(n_neurons)    
        
#         pre_activation_list = []
#         label_list = []
#         prediction_list = []

#         total = 0

#         for images, labels, idx in loader:
#             total += labels.size(0)

#             images, labels = images.to(device), labels.to(device)  
#             logit, pre_activation = model(images, idx, return_intermediates=True, neuron_indexes=neurons_indexes, layer_indexes = layer_indexes)
#             _, predictions = torch.max(logit, 1)

#             pre_activation_list.append(pre_activation.detach())
#             label_list.append(labels.detach())
#             prediction_list.append(predictions.detach())

#             for ii in range(args.num_classes):
#                 if getattr(args, "CND_well_classified_filter", True):
#                     mask = (labels==ii)&(predictions==labels) # select only well predicted neurons
#                 else:
#                     mask = (labels==ii)
                    
#                 if mask.sum() != 0:
#                     neuron_activations = pre_activation[mask, :].detach().cpu()

#                     # max value update
#                     max_values = torch.max(neuron_activations, axis=0)[0]
#                     mask_max_value = max_values > max_activation
#                     max_activation[mask_max_value] = max_values[mask_max_value]

#                     # min value update
#                     min_values = torch.min(neuron_activations, axis=0)[0]
#                     mask_min_value = min_values < min_activation
#                     min_activation[mask_min_value] = min_values[mask_min_value]


#                     if ii in neuron_activations_dict: 
#                         neuron_activations_dict[ii] = torch.cat([neuron_activations_dict[ii], neuron_activations], axis=0)
#                     else:
#                         neuron_activations_dict[ii] = neuron_activations    

#             if getattr(args, "debug_reduced_epoch_duration", None):  # Default to None if not present
#                 if total > args.debug_reduced_epoch_duration:
#                     break
        
#         pre_activations = torch.cat(pre_activation_list, dim=0).detach().cpu()
#         labels = torch.cat(label_list).detach().cpu()
#         predictions = torch.cat(prediction_list).detach().cpu()

#         if len(neuron_activations_dict)>0:

#             if not isinstance(args.CND_type, list):
#                 args.CND_type = [args.CND_type]

#             for CND_type in args.CND_type:

#                 if CND_type == "CND_mean":
#                     CND = mean_approx_JSD(neuron_activations_dict, n_neurons, max_activation, args)

#                 elif CND_type == "CND_PMF":
#                     CND = jsd_bins_hist(neuron_activations_dict, n_neurons, max_activation, min_activation, args)

#                 elif CND_type == "CND_kNN":
#                     CND = multi_distr_jsd_kNN(neuron_activations_dict, n_neurons, max_activation, min_activation, args)

#                 elif CND_type == "CND_KDE":
#                     CND = jsd_KDE(pre_activations, labels, args)

#                 elif CND_type == "CND_GL":
#                     CND = jsd_GaussianLike(pre_activations, labels, args)

#                 elif "top" in CND_type:
#                     CND = topK_jsd_bins_hist_approx(neuron_activations_dict, n_neurons, max_activation, min_activation, args, CND_type=CND_type)

#                 performances_dict[CND_type] = CND
#                 logger.info(f"{CND_type} -- {torch.mean(CND)}")
#         # else:
#         #     CND = torch.zeros(n_neurons)
#         #     print("CND NOT COMPUTED")
          

#     return performances_dict
    
