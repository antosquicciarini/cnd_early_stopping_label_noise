import torch
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from performances import Performances
from critical_sample_ratio import compute_CSR
from prediction_changes import compute_changed_predictions
from graph_entropy import graph_entropy_evaluation
from loss_sesivity import loss_sensivity
from neuron_frequency_activation import neuron_frequency_activation, neuron_frequency_activation_plot
from cnd import cnd
import numpy as np
from torch.cuda.amp import GradScaler, autocast
import time
from loss_regularizer import apply_regularizers
import logging
import os
import sys
import argparse
from torch.utils.data import Dataset, DataLoader, Subset
import random
from cnd import calculate_neuron_pre_activations, compute_cnd_metrics
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu
import pandas as pd
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy

# Import your project-specific modules
from network_structure import model_definition
from model_trainining_and_evaluation import train_and_evaluate_model
from load_dataset import load_dataset
from parameter_settings import parameter_settings
import logging
from model_trainining_and_evaluation import model_evaluate

from scipy.stats import entropy

def compute_differential_entropy(pre_activations, method="trapz"):
    """
    Computes the differential entropy of the pre-activations using a given method.

    Parameters:
    - pre_activations (torch.Tensor): Tensor of shape (samples, neurons) with pre-activation values.
    - method (str): The method to compute differential entropy. Currently supports "trapz".

    Returns:
    - torch.Tensor: Differential entropy values for each neuron.
    """
    entropies = []
    for neuron_idx in range(pre_activations.shape[1]):
        neuron_values = pre_activations[:, neuron_idx].cpu().numpy()

        # Estimate the PDF using a histogram
        hist, bin_edges = np.histogram(neuron_values, bins=50, density=True)

        # Compute bin centers and entropy using trapezoidal approximation
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        if method == "trapz":
            # Ensure non-zero probabilities for entropy computation
            pdf = hist / np.sum(hist)
            pdf = np.clip(pdf, 1e-12, None)  # Avoid log(0)
            ent = -np.trapz(pdf * np.log(pdf), bin_centers)
        else:
            raise ValueError("Unsupported method for entropy computation")

        entropies.append(ent)

    return torch.tensor(entropies, dtype=torch.float32)


def stat_significant_neuron_act(model, threshold, loader):
    # compute neurons pre-act
    # for each neuron compute stat tests, 
    # gather the results based on the layer
    return None

def stat_significant_cnd(model, threshold, loader):
    # compute the cnd over the dataset
    # see if there is a signinficant difference between layers
    return None

def print_evaluation_results(test_mask_acc, corrupted_mask_acc, no_corrupted_mask_acc):
    """
    Prints the evaluation results in a formatted way.

    Parameters:
    - fixed_no_mask_acc, test_no_mask_acc, corrupted_no_mask_acc, no_corrupted_no_mask_acc: 
      Accuracies without the CND-based mask applied.
    - fixed_mask_acc, test_mask_acc, corrupted_mask_acc, no_corrupted_mask_acc: 
      Accuracies with the CND-based mask applied.
    """
    print("Evaluation Results:")
    print("====================")
    print(f"  Test Dataset Accuracy:        {test_mask_acc:.4f}")
    print(f"  Non-Corrupted Dataset Accuracy: {no_corrupted_mask_acc:.4f}")
    print(f"  Corrupted Dataset Accuracy:   {corrupted_mask_acc:.4f}")
    print(f"  Expected Corrupted Dataset Accuracy:   {((1 - test_mask_acc) / 9):.4f}")


def convert_last_layer_mask(mask_dict):
    """
    Converts the mask dictionary to apply the mask only at the last layer.
    All other layers will have a mask of all `True`.

    Parameters:
    - mask_dict (dict): The original mask dictionary with layer-specific masks.

    Returns:
    - dict: Updated mask dictionary where only the last layer's mask is retained.
    """
    last_layer = max(mask_dict.keys())
    last_mask = mask_dict[last_layer]
    converted = {key: torch.ones_like(value, dtype=torch.bool) for key, value in mask_dict.items()}
    converted[last_layer] = last_mask
    return converted

def layer_mask(mask_dict, layer_list):
    
    """
    Converts the mask dictionary to apply the mask only at the last layer.
    All other layers will have a mask of all `True`.

    Parameters:
    - mask_dict (dict): The original mask dictionary with layer-specific masks.

    Returns:
    - dict: Updated mask dictionary where only the last layer's mask is retained.
    """

    mask_dict_copy = copy.deepcopy(mask_dict)

    for layer_indx in layer_list:
        mask_dict_copy[layer_indx] = torch.ones_like(mask_dict[layer_indx], dtype=torch.bool)
    return mask_dict_copy


def mask_eval(model, thresh, threshold_dir, loaders, device, args, eval_measure="diff_entropy", global_mask=False):
    """
    Evaluate Conditional Neuron Divergence (CND) metrics for a given model and dataset loader.
    Also evaluate with random, last-layer, and penultimate-layer masks.

    Parameters:
    - loaders (dict): Dataset loaders.
    - model (nn.Module): Model to evaluate.
    - device (torch.device): Device to use (e.g., 'cuda').
    - args (argparse.Namespace): Configuration arguments.

    Returns:
    - dict: CND evaluation metrics.
    """
    model.eval()
    model.disable_mask()
    results = {}

    num_layers = len(args.neurs_x_hid_lyr)  # Retrieve the number of layers
    all_layer_idxs = list(range(num_layers))  # Adaptable list of all layer indices

    with torch.no_grad():
        # Compute neuron pre-activations and predictions
        pre_act, labels, preds = calculate_neuron_pre_activations(loaders["fixed"], model, device, args, max_images=70000)

        # Filter correctly classified samples
        correct_mask = labels == preds
        pre_act = pre_act[correct_mask]
        labels = labels[correct_mask]

        if eval_measure == "cnd":
            cnd_type = "CND_PMF"
            filter_values = compute_cnd_metrics(cnd_type, pre_act, labels, args)
        elif eval_measure == "diff_entropy":
            filter_values = compute_differential_entropy(pre_act, method="trapz")

        masks = {}
        random_masks = {}

        for key, val in args.neurs_x_hid_lyr.items():
            # Compute cumulative neurons up to the current layer
            cum_neurs = torch.tensor(list(args.neurs_x_hid_lyr.values()))
            start_idx = torch.sum(cum_neurs[:key])

            # Compute CND mask for the current layer
            layer_vals = filter_values[start_idx: start_idx + val]

            if threshold_dir == "lower":
                if thresh == 0:
                    mask = torch.zeros_like(layer_vals, dtype=torch.bool)
                elif global_mask:
                    mask = layer_vals < torch.quantile(filter_values, thresh)
                else:
                    mask = layer_vals < torch.quantile(layer_vals, thresh)

            elif threshold_dir == "higher":
                if thresh == 0:
                    mask = torch.ones_like(layer_vals, dtype=torch.bool)
                elif global_mask:
                    mask = layer_vals > torch.quantile(filter_values, thresh)
                else:
                    mask = layer_vals > torch.quantile(layer_vals, thresh)

            masks[key] = mask.to(device)

            # Generate a random mask with the same number of active neurons
            num_active = mask.sum().item()
            rand_mask = torch.zeros_like(mask, dtype=torch.bool)
            rand_idxs = torch.randperm(mask.size(0))[:num_active]
            rand_mask[rand_idxs] = True
            random_masks[key] = rand_mask.to(device)

        acc_mode = True

        # Apply masks and evaluate for different settings
        def evaluate_and_store(mask_dict, key_prefix):
            model.set_mask(mask_dict)
            results[f"{key_prefix}_test_acc"], _ = model_evaluate(model, loaders["test"], device, args, apply_mask=True, stop_criteria_enabled=acc_mode)
            results[f"{key_prefix}_corrupt_acc"], _ = model_evaluate(model, loaders["corrupted"], device, args, apply_mask=True, stop_criteria_enabled=acc_mode)
            results[f"{key_prefix}_clean_acc"], _ = model_evaluate(model, loaders["non_corrupted"], device, args, apply_mask=True, stop_criteria_enabled=acc_mode)

        # Full mask evaluation
        evaluate_and_store(masks, "mask")

        # Random mask evaluation
        evaluate_and_store(random_masks, "mask_rand")

        # Last layer mask evaluation
        last_layer_masks = layer_mask(masks, all_layer_idxs[:-1])  # Turn off all but the last layer
        evaluate_and_store(last_layer_masks, "mask_last")

        last_layer_rand_masks = layer_mask(random_masks, all_layer_idxs[:-1])  # Random for the last layer
        evaluate_and_store(last_layer_rand_masks, "mask_last_rand")

        # Penultimate layer mask evaluation
        penultimate_layer_masks = layer_mask(masks, all_layer_idxs[:-2])  # Turn off all but the penultimate layer
        evaluate_and_store(penultimate_layer_masks, "mask_penultimate")

        penultimate_layer_rand_masks = layer_mask(random_masks, all_layer_idxs[:-2])  # Random for the penultimate layer
        evaluate_and_store(penultimate_layer_rand_masks, "mask_penultimate_rand")

        # Print results
        for key, prefix in [("mask", "CND Mask"), ("mask_rand", "Random Mask"), ("mask_last", "Last-Layer Mask"),
                            ("mask_last_rand", "Last-Layer Random Mask"), ("mask_penultimate", "Penultimate-Layer Mask"),
                            ("mask_penultimate_rand", "Penultimate-Layer Random Mask")]:
            print(f"\nEvaluation with {prefix}:")
            print_evaluation_results(results[f"{key}_test_acc"], results[f"{key}_corrupt_acc"], results[f"{key}_clean_acc"])

    return results

# ----------------------- Utility Functions -----------------------
def save_json(data, file_path):
    """Saves a dictionary to a JSON file."""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to: {file_path}")


def compute_p_values(pre_activations, labels, unique_classes):
    """
    Computes p-values using Kruskal-Wallis H-test for each neuron.
    """
    p_values = []
    for neuron_idx in range(pre_activations.shape[1]):
        groups = [pre_activations[labels == cls, neuron_idx] for cls in unique_classes]
        _, p_value = kruskal(*groups)
        p_values.append(p_value)
    return np.array(p_values)


def bonferroni_correction(p_values, num_tests):
    """
    Applies Bonferroni correction to the p-values.
    """
    adjusted_p_values = p_values * num_tests
    adjusted_p_values[adjusted_p_values > 1] = 1
    return adjusted_p_values


def compute_layer_significance(filter_values, args):
    """
    Computes the significance of CND values per layer using a Mann-Whitney U test.
    """
    layer_significance = {}
    for key, val in args.neurs_x_hid_lyr.items():
        cum_neurs = torch.tensor(list(args.neurs_x_hid_lyr.values()))
        start_idx = torch.sum(cum_neurs[:key])

        inside_range = filter_values[start_idx: start_idx + val]
        outside_range = torch.cat((filter_values[:start_idx], filter_values[start_idx + val:]))

        stat, p_value = mannwhitneyu(inside_range.cpu().numpy(), outside_range.cpu().numpy(), alternative='two-sided')
        layer_significance[key] = p_value

        significance_msg = (
            f"Statistical difference detected between ranges (p = {p_value:.4f})."
            if p_value < 0.05 else
            f"No statistical difference detected between ranges (p = {p_value:.4f})."
        )
        print(significance_msg)

    return layer_significance

def mean_std_x_layer(filter_values, args):
    """
    Computes the significance of CND values per layer using a Mann-Whitney U test.
    """
    cnd_x_layer_mean = {}
    cnd_x_layer_std = {}

    for key, val in args.neurs_x_hid_lyr.items():
        cum_neurs = torch.tensor(list(args.neurs_x_hid_lyr.values()))
        start_idx = torch.sum(cum_neurs[:key])

        inside_range = filter_values[start_idx: start_idx + val]
        outside_range = torch.cat((filter_values[:start_idx], filter_values[start_idx + val:]))

        cnd_mean = torch.mean(inside_range)
        cnd_std = torch.std(inside_range)
        cnd_x_layer_mean[key] = cnd_mean
        cnd_x_layer_std[key] = cnd_std

    return cnd_x_layer_mean, cnd_x_layer_std

# ----------------------- Core Functions -----------------------
def create_dataframe_short(cnd_x_layer_mean, cnd_x_layer_std, layer_significance, sign_neurons_per_layer, args):
    """
    Creates a pandas DataFrame with shorter column names from CND-related metrics and expand_dataset.

    Args:
        cnd_x_layer_mean (dict): Mean values of CND per layer.
        cnd_x_layer_std (dict): Standard deviation of CND per layer.
        layer_significance (dict): Significance per layer.
        sign_neurons_per_layer (dict): Proportion of significant neurons per layer.
        args (argparse.Namespace): Arguments containing dataset, label noise, and file name.

    Returns:
        pd.DataFrame: DataFrame containing the structured information.
    """
    # Prepare the data

    pickle_file_path = f"{args.results_dir}/performances.pkl"
    # Load the pickle file
    with open(pickle_file_path, "rb") as file:
        performance = pickle.load(file)

    test_acc = performance.test_acc[-1].item()  # Assuming 'test_acc' is stored in performances
    #train_loss = performances.train_loss[-1]
    misslab_acc = performance.misslab_acc[-1]

    rows = []
    for layer, mean in cnd_x_layer_mean.items():
        rows.append({
            "Dataset": args.dataset,
            "Noise": args.symmetric_label_noise_ratio if hasattr(args, 'symmetric_label_noise_ratio') else args.noise_type,            
            "File": args.filename,
            "dropout": getattr(args, "dropout_rate", 0.0),
            "Layer_Index": layer,
            "Layer_Sig": layer_significance.get(layer, None),
            "Sig_Neurons": sign_neurons_per_layer.get(layer, None),
            "cnd_mean": mean.item() if isinstance(mean, torch.Tensor) else mean,
            "cnd_std": cnd_x_layer_std[layer].item() if isinstance(cnd_x_layer_std[layer], torch.Tensor) else cnd_x_layer_std[layer],
            "test_acc": test_acc,
            "misslab_acc": misslab_acc,

        })

    # Convert rows to DataFrame
    df = pd.DataFrame(rows)
    return df

def statistical_test_activations(model, loaders, model_dir, device, args):
    """
    Perform statistical tests on neuron activations and save the results.
    """
    model.eval()
    model.disable_mask()

    with torch.no_grad():
        pre_act, labels, preds = calculate_neuron_pre_activations(loaders["fixed"], model, device, args)

        #TOTEST
        # # Generate a random label tensor with the same shape as the original labels
        # num_classes = labels.max().item() + 1  # Assuming labels are 0-indexed
        # random_labels = torch.randint(0, num_classes, labels.shape, device=labels.device)
        # # Replace the labels tensor with the random labels
        # labels = random_labels

        # Convert to numpy for statistical testing
        pre_act_np = pre_act.numpy()
        labels_np = labels.numpy()
        unique_classes = np.unique(labels_np)

        # Compute p-values for activations
        p_values = compute_p_values(pre_act_np, labels_np, unique_classes)
        adjusted_p_values = bonferroni_correction(p_values, pre_act_np.shape[1])

        # Map significant neurons per layer
        sign_neurons_per_layer = {}
        for key, val in args.neurs_x_hid_lyr.items():
            cum_neurs = torch.tensor(list(args.neurs_x_hid_lyr.values()))
            start_idx = torch.sum(cum_neurs[:key])
            layer_p_values = adjusted_p_values[start_idx: start_idx + val]
            significant_neurons = np.where(layer_p_values < 0.05)[0]
            sign_neurons_per_layer[key] = len(significant_neurons) / len(layer_p_values)

            print(f"Number of significant neurons in layer {key}: {sign_neurons_per_layer[key]:.2%}")

        # Save significant neurons per layer
        #save_json(sign_neurons_per_layer, os.path.join(model_dir, "significant_neurons_per_layer.json"))

        # Compute layer-wise CND significance
        filter_values = compute_cnd_metrics("CND_PMF", pre_act, labels, args)
        layer_significance = compute_layer_significance(filter_values, args)

        # Save layer significance results
        #save_json(layer_significance, os.path.join(model_dir, "layer_significance_p_values.json"))

        cnd_x_layer_mean, cnd_x_layer_std  = mean_std_x_layer(filter_values, args) 
        df = create_dataframe_short(cnd_x_layer_mean, cnd_x_layer_std, layer_significance, sign_neurons_per_layer, args)


        return df



def find_model_folder(base_dir, target_folder_name):
    """
    Searches for a folder with the given name in all subdirectories of the base directory.

    Parameters:
    - base_dir (str): The base directory to start the search.
    - target_folder_name (str): The name of the folder to search for.

    Returns:
    - list: A list of paths where the folder is found.
    """
    matching_paths = []
    for root, dirs, files in os.walk(base_dir):
        if target_folder_name in dirs:
            return os.path.join(root, target_folder_name)
    return None


def upload_saved_model(model_name):#model_class, optimizer_class, model_path, device, lr=0.001):
    """
    Upload a saved PyTorch model and its optimizer state.

    Parameters:
    - model_class: The class of the model to instantiate (e.g., ResNet50).
    - optimizer_class: The optimizer class (e.g., torch.optim.SGD).
    - model_path: Path to the saved model file.
    - device: The device to load the model onto (e.g., 'cpu' or 'cuda').
    - lr: Learning rate for the optimizer if state is restored.

    Returns:
    - model: The loaded model ready for evaluation.
    - optimizer: The restored optimizer with state.
    - args: The saved arguments for reproducibility.
    """

    device = torch.device("mps")
    folder_path = find_model_folder("models", model_name)

    model_path = f"{folder_path}/{model_name}_model.pth"
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    params = checkpoint.get('args', None)
    args = argparse.Namespace(verbose=False, verbose_1=False)
    for k, v in params.items():
        setattr(args, k, v)

    model, args = model_definition(device, args)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode
    
    print(f"Model and optimizer state loaded from: {model_path}")

    return model, device, args

def plot_absolute_and_relative(thresholds, threshold_dir, results, model_dir, model_name, eval_measure="cnd", global_mask=False):
    """
    Plots absolute and relative accuracy changes for each group of results and saves them in a folder.

    Parameters:
    - thresholds: List of thresholds used for evaluation.
    - threshold_dir: Direction of the threshold ("lower" or "higher").
    - results: List of dictionaries, each containing accuracy metrics.
    - model_dir: Directory to save the plots.
    - model_name: Name of the model being evaluated.
    - eval_measure: Evaluation measure used (default is "cnd").
    """

    # Create a subfolder for saving images
    image_dir = os.path.join(model_dir, f"{model_name}_plots")
    os.makedirs(image_dir, exist_ok=True)

    # Extract accuracy metrics from results
    groups = {
        "mask": {
            "test": [res['mask_test_acc'] for res in results],
            "corrupt": [res['mask_corrupt_acc'] for res in results],
            "clean": [res['mask_clean_acc'] for res in results],
        },
        "mask_rand": {
            "test": [res['mask_rand_test_acc'] for res in results],
            "corrupt": [res['mask_rand_corrupt_acc'] for res in results],
            "clean": [res['mask_rand_clean_acc'] for res in results],
        },
        "mask_last": {
            "test": [res['mask_last_test_acc'] for res in results],
            "corrupt": [res['mask_last_corrupt_acc'] for res in results],
            "clean": [res['mask_last_clean_acc'] for res in results],
        },
        "mask_last_rand": {
            "test": [res['mask_last_rand_test_acc'] for res in results],
            "corrupt": [res['mask_last_rand_corrupt_acc'] for res in results],
            "clean": [res['mask_last_rand_clean_acc'] for res in results],
        },
        "mask_penultimate": {
            "test": [res['mask_penultimate_test_acc'] for res in results],
            "corrupt": [res['mask_penultimate_corrupt_acc'] for res in results],
            "clean": [res['mask_penultimate_clean_acc'] for res in results],
        },
        "mask_penultimate_rand": {
            "test": [res['mask_penultimate_rand_test_acc'] for res in results],
            "corrupt": [res['mask_penultimate_rand_corrupt_acc'] for res in results],
            "clean": [res['mask_penultimate_rand_clean_acc'] for res in results],
        },
    }
    
    for group_name, group_data in groups.items():
        plot_name = f"abs_{eval_measure}_{threshold_dir}_{group_name.replace(' ', '_')}_global_mask_{global_mask}"
        # Absolute values plot
        plt.figure(figsize=(10, 5))
        for metric_name, values in group_data.items():
            if threshold_dir == "lower":
                # Invert the order of the vector if threshold_dir is "lower"
                values = values[::-1]
            plt.plot(thresholds, values, label=f"{metric_name.capitalize()} Accuracy", marker='o')

        plt.title(plot_name)
        plt.xlabel("Threshold")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid()
        # Save the plot
        absolute_plot_path = os.path.join(image_dir, plot_name)
        plt.savefig(absolute_plot_path)
        plt.close()

        # Relative changes plot
        plot_name = f"rel_{eval_measure}_{threshold_dir}_{group_name.replace(' ', '_')}_global_mask_{global_mask}"
        plt.figure(figsize=(10, 5))
        for metric_name, values in group_data.items():
            relative_changes = [(v / values[0] - 1) if values[0] != 0 else 0 for v in values]
            if threshold_dir == "lower":
                relative_changes = relative_changes[::-1]
            plt.plot(thresholds, relative_changes, label=f"{metric_name.capitalize()} Accuracy", marker='o')
        plt.title(plot_name)
        plt.xlabel("Threshold")
        plt.ylabel("Relative Change")
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # Reference line at 0
        plt.legend()
        plt.grid()
        # Save the plot
        relative_plot_path = os.path.join(image_dir, plot_name)
        plt.savefig(relative_plot_path)
        plt.close()



def parallel_threshold_computation(model, thresholds, threshold_dir, loaders, device, args, eval_measure, global_mask):
    """
    Parallelize the threshold computation.

    Parameters:
    - model: The PyTorch model being evaluated.
    - thresholds: List of thresholds to compute.
    - threshold_dir: Direction of the threshold ("lower" or "higher").
    - loaders: Dictionary of dataset loaders.
    - device: The device to use for computation.
    - args: Additional arguments for evaluation.
    - eval_measure: Evaluation measure used.
    - global_mask: Boolean flag for global masking.

    Returns:
    - List of results corresponding to each threshold.
    """
    results = []

    # Define the task for each threshold
    def compute_threshold(threshold):
        print(f"Starting computation for threshold {threshold}...")
        result = mask_eval(model, threshold, threshold_dir, loaders, device, args, eval_measure=eval_measure, global_mask=global_mask)
        print(f"Finished computation for threshold {threshold}.")
        return result

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=min(len(thresholds), os.cpu_count())) as executor:
        # Submit all tasks
        future_to_threshold = {executor.submit(compute_threshold, t): t for t in thresholds}

        # Collect results as they are completed
        for future in as_completed(future_to_threshold):
            threshold = future_to_threshold[future]
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Threshold {threshold} raised an exception: {e}")

    return results

def main_evaluation(model_name, thresholds, apply_mask=False, global_mask=False):
    """
    Main function to evaluate a pre-trained model.
    """
    result = []

    # Adjust working directory
    if "program" in os.getcwd():
        os.chdir("..")

    #Model update
    model, device, args = upload_saved_model(model_name)

    #NEW SETTINGS
    args.parallelize_mask = False  # Set to False to disable parallelization

    print(f"EVALUATING {args.filename} ...")

    # Fix the random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Data preparation
    train_dataset, test_dataset, corrupted_samples, not_corrupted_samples, args = load_dataset(args) 

    train_loader_fixed = DataLoader(train_dataset, batch_size=args.fixed_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.fixed_batch_size, shuffle=False)

    # Corrupted and non-corrupted loaders
    train_loader_corrupted = DataLoader(Subset(train_dataset, corrupted_samples),
                                         batch_size=args.fixed_batch_size, shuffle=False)

    train_loader_not_corrupted = DataLoader(Subset(train_dataset, not_corrupted_samples), 
                                            batch_size=args.fixed_batch_size, shuffle=False)

    # Loaders for different datasets
    loaders = {
        "test": test_loader,  # Replace with your test data loader
        "fixed": train_loader_fixed,  # Replace with your fixed data loader
        "corrupted": train_loader_corrupted,  # Optional
        "non_corrupted": train_loader_not_corrupted  # Optional
    }

    # Statistical test
    eval_criteria = "cnd" #"diff_entropy"
    df = statistical_test_activations(model, loaders, args.results_dir, device, args)
    threshold_dir_list = ["higher"] #L or H (lower or higher)

    if apply_mask:

       
        for threshold_dir in threshold_dir_list:

            if not args.parallelize_mask:
                results = []
                for treshold in thresholds:
                    results.append(mask_eval(model, treshold, threshold_dir, loaders, device, args, eval_measure=eval_criteria, global_mask=global_mask))
                    print(f"Computing threshold {treshold} ...")
                    print(f"Result threshold {treshold} DONE")
                    
                plot_absolute_and_relative(thresholds, threshold_dir, results, args.results_dir, model_name, eval_measure=eval_criteria, global_mask=global_mask)
            
            else:

                print(f"Parallelizing computations for thresholds in {threshold_dir} direction...")
                results = parallel_threshold_computation(
                    model, thresholds, threshold_dir, loaders, device, args,
                    eval_measure=eval_criteria, global_mask=global_mask
                )
                plot_absolute_and_relative(thresholds, threshold_dir, results, args.results_dir, model_name, eval_measure=eval_criteria, global_mask=global_mask)
            
    return df

if __name__ == "__main__":

    ### SETTINGS ###
    model_list = ['exp_2024-12-31_16-50_CIFAR10_baseline_sim_0'] #['exp_2024-12-31_16-50_CIFAR10_baseline_sim_0 exp_2024-12-20_14-31_MNIST_FashMNIST_baseline_sim_0 exp_2024-12-20_14-31_MNIST_FashMNIST_baseline_sim_1']#['exp_2024-12-19_15-44_MNIST_FashMNIST_baseline_sim_1']     #['exp_2024-12-20_15-16_CIFAR10_FCN_baseline_sim_1']    #['exp_2024-12-19_15-44_MNIST_FashMNIST_baseline_sim_1']    
    experiment = "analysis_baseline"
    save_global_results = True
    apply_mask = True
    thresholds = [0.0, 0.005, 0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0] #[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7] #[0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7]
    global_mask = False

    # Call the evaluation
    df_list = []
    for model_name in model_list:
        df_list.append(main_evaluation(model_name, thresholds, apply_mask=apply_mask, global_mask=global_mask))
    df = pd.concat(df_list, ignore_index=True)
    output_file = f"results/{experiment}.xlsx"  # Specify the desired output file path
    df.to_excel(output_file, index=False)

    if save_global_results:

        # Group the DataFrame and calculate the mean
        grouped_df = df.groupby(["File", "Dataset", "Noise", "dropout", "Layer_Index", "test_acc", "misslab_acc"])[["Sig_Neurons", "cnd_mean", "cnd_std"]].mean().reset_index()

        # Save the grouped DataFrame to an Excel file
        grouped_output_file = f"results/{experiment}_grouped.xlsx"  # Specify the desired output file path
        grouped_df.to_excel(grouped_output_file, index=False)

    # Pivoting the data to create multi-index columns
    pivoted = grouped_df.pivot_table(
        index=["Dataset","dropout", "Noise"],
        columns="Layer_Index",
        values=["cnd_mean", "cnd_std", "Sig_Neurons"]
    )

    # Adding test_acc, misslab_acc, and Sig_Neurons as additional columns
    additional_metrics = grouped_df.groupby(["Dataset", "dropout", "Noise"]).agg({
        "test_acc": "first",
        "misslab_acc": "first"
    })

    # Round all relevant numeric values to two significant digits
    pivoted = pivoted.applymap(lambda x: round(x, 2) if pd.notnull(x) else x)
    additional_metrics = additional_metrics.applymap(lambda x: round(x, 2) if pd.notnull(x) else x)

    # Combine pivoted data with additional metrics
    final_table = pd.concat([pivoted, additional_metrics], axis=1)

    # Formatting table in LaTeX format
    latex_table = final_table.to_latex(
        multirow=True,
        float_format="%.4f",
        caption="Results with Layer-wise CND Mean and Std along with Additional Metrics",
        label="tab:results",
        longtable=True
    )
    # Saving the LaTeX table to a file
    with open(f"results/{experiment}_grouped.tex" , "w") as f:
        f.write(latex_table)


    print(f"Grouped DataFrame saved to {grouped_output_file}")