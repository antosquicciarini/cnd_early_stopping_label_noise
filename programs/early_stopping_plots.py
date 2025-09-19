"""Plot early stopping proxy metrics (CND, KPA, PC) against test accuracy for specified experiments.
Allows command-line configuration of experiment directory, metrics, windows, and filtering.
"""


# "exp_2025_02_13_05_49_58_early_stopping_CIFAR10_real_noise_sim_20", #CIFAR10 Agree
# "exp_2025_02_12_14_50_52_early_stopping_CIFAR10_symmetric_sim_0", #CIFAR10 10
# "exp_2025_02_12_19_17_06_early_stopping_CIFAR10_real_noise_sim_6", #CIFAR10 rand=1
# "exp_2025_02_13_08_08_57_early_stopping_CIFAR100_symmetric_sim_10", #CIFAR10 Agree
# "exp_2025_02_12_21_37_48_early_stopping_CIFAR10_symmetric_sim_9",


# # #CIFAR100
# "exp_2025_05_05_14_00_15_early_stopping_CIFAR100_symmetric_preact_sim_0", #10%
# "exp_2025_05_05_15_18_53_early_stopping_CIFAR100_symmetric_preact_sim_1", #20%
# "exp_2025_05_05_18_13_52_early_stopping_CIFAR100_symmetric_preact_sim_3", #40%
# "exp_2025_05_05_13_55_58_early_stopping_CIFAR100_real_noise_preact_sim_0", #worse


# # #CIFAR10
# "exp_2025_05_05_13_26_46_early_stopping_CIFAR10_symmetric_preact_sim_0",#10
# "exp_2025_05_05_14_35_16_early_stopping_CIFAR10_symmetric_preact_sim_1", #20
# "exp_2025_05_05_15_44_06_early_stopping_CIFAR10_symmetric_preact_sim_2", #40
# "exp_2025_05_05_18_15_11_early_stopping_CIFAR10_symmetric_preact_sim_4", #80

# "exp_2025_05_05_16_32_07_early_stopping_CIFAR10_real_noise_preact_sim_4", # agree
# "exp_2025_05_05_12_56_40_early_stopping_CIFAR10_real_noise_preact_sim_1", #rand1
# "exp_2025_05_05_14_07_46_early_stopping_CIFAR10_real_noise_preact_sim_2", #rand2
# "exp_2025_05_05_11_39_18_early_stopping_CIFAR10_real_noise_preact_sim_0" #worse


# # NEWS - NEWS_CND_yu19_main_exp_adjs
# "exp_2025_06_23_17_44_48_NEWS_CND_yu19_main_exp_sim_0"
# "exp_2025_06_23_17_58_05_NEWS_CND_yu19_main_exp_sim_5"
# "exp_2025_06_23_18_10_21_NEWS_CND_yu19_main_exp_sim_10"
# "exp_2025_06_23_18_21_30_NEWS_CND_yu19_main_exp_sim_15"
# "exp_2025_06_23_18_31_28_NEWS_CND_yu19_main_exp_sim_20'


# # NEWS - NEWS_CND_yu19_main_exp_adjs
# "exp_2025_06_23_17_44_48_NEWS_CND_yu19_main_exp_sim_0"
# "exp_2025_06_23_17_58_05_NEWS_CND_yu19_main_exp_sim_5"
# "exp_2025_06_23_18_10_21_NEWS_CND_yu19_main_exp_sim_10"
# "exp_2025_06_23_18_21_30_NEWS_CND_yu19_main_exp_sim_15"
# "exp_2025_06_23_18_31_28_NEWS_CND_yu19_main_exp_sim_20'



import os
import json
import pickle
import argparse
from pathlib import Path
import random
import re
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from PIL import Image

from load_info import extract_experiment_features, moving_average
from network_structure import model_definition

CND_TYPE = "PMF"
EXPERIMENT_DIRECTION = "early_stopping_variab_no_early_stopping_preact_fin_no_mil" 
#early_stopping_variab_no_early_stopping_preact_fin_no_mil
#NEWS_CND_yu19_main_dropout_final

# Define experiments and windows to plot
EXPERIMENTS_TO_PLOT = [ 

    # # NEWS
    # "exp_2025_07_21_22_56_51_NEWS_CND_dropout_sim_0",
    # "exp_2025_07_21_23_45_34_NEWS_CND_dropout_sim_5",
    # "exp_2025_07_22_00_31_37_NEWS_CND_dropout_sim_10",
    # "exp_2025_07_22_01_15_51_NEWS_CND_dropout_sim_15"

    # #CIFAR100
    "exp_2025_05_05_14_00_15_early_stopping_CIFAR100_symmetric_preact_sim_0", #10%
    "exp_2025_05_05_15_18_53_early_stopping_CIFAR100_symmetric_preact_sim_1", #20%
    "exp_2025_05_05_18_13_52_early_stopping_CIFAR100_symmetric_preact_sim_3", #40%
    "exp_2025_05_05_13_55_58_early_stopping_CIFAR100_real_noise_preact_sim_0", #worse


    # #CIFAR10
    "exp_2025_05_05_13_26_46_early_stopping_CIFAR10_symmetric_preact_sim_0",#10
    "exp_2025_05_05_14_35_16_early_stopping_CIFAR10_symmetric_preact_sim_1", #20
    "exp_2025_05_05_15_44_06_early_stopping_CIFAR10_symmetric_preact_sim_2", #40
    "exp_2025_05_05_18_15_11_early_stopping_CIFAR10_symmetric_preact_sim_4", #80

    "exp_2025_05_05_16_32_07_early_stopping_CIFAR10_real_noise_preact_sim_4", # agree
    "exp_2025_05_05_12_56_40_early_stopping_CIFAR10_real_noise_preact_sim_1", #rand1
    "exp_2025_05_05_14_07_46_early_stopping_CIFAR10_real_noise_preact_sim_2", #rand2
    "exp_2025_05_05_11_39_18_early_stopping_CIFAR10_real_noise_preact_sim_0" #worse

    ]

WINDOWS_TO_PLOT = [5]

def parse_args():
    parser = argparse.ArgumentParser(description="Plot early stopping metrics vs test accuracy")
    parser.add_argument("--experiment-direction", default=EXPERIMENT_DIRECTION,
                        help="Subdirectory under models to load experiments from")
    
    parser.add_argument("--cnd-type", default=CND_TYPE, help="CND type to slice")

    parser.add_argument("--experiments", nargs="+", default=EXPERIMENTS_TO_PLOT,
                        help="List of experiment names to plot")
    
    parser.add_argument("--windows", nargs="+", type=int, default=WINDOWS_TO_PLOT,
                        help="Window sizes for moving averages")
    
    parser.add_argument("--filter-expand-dataset", action="store_true",
                        help="Filter experiments where expand_dataset=True")
    
    parser.add_argument(
        "--cnd-percentile",
        default=90.0,
        help="Percentile (0-100) for selecting the CND quantile (e.g., 25 for the first quartile)",
    )
    return parser.parse_args()

# Detect device: prefer CUDA, then Apple MPS, then CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def save_plot_with_path(exp_name: str, experiment_direction: str):
    """
    Save the current matplotlib plot to the appropriate path and filename, using logic
    similar to the original _save_plot_with_path.
    """
    parts = exp_name.split("_")
    if "CIFAR10" in exp_name or "CIFAR100" in exp_name:
        dataset = parts[5]
        noise_type = parts[6]
        noise_level = parts[-1]  # e.g., sim_2
        title_exp = f"{exp_name} ({noise_type}, {noise_level})"
    else:
        title_exp = exp_name
    save_exp_name = title_exp.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    plots_dir = Path("models") / experiment_direction / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / f"{save_exp_name}.png")
    plt.close()

def new_performances_plot(test_acc, CND, KPA, PC, windows, exp_name):
    def normalize_data(data):
        data_min, data_max = torch.min(data), torch.max(data)
        if data_max - data_min == 0:
            return torch.zeros_like(data)
        return (data - data_min) / (data_max - data_min)

    # Increased figure size for readability
    fig, ax1 = plt.subplots(figsize=(18, 8))
    ax2 = ax1.twinx()

    # Plot test accuracy on right y-axis, with lower zorder so legend overlays on top
    ax2.plot(test_acc, label="Test Accuracy", linestyle="-", linewidth=3.5, color="black", zorder=1)

    # Colors for different metrics
    colors = {"CND": "blue", "KPA": "green", "PC": "red"}

    for i, w in enumerate(sorted(windows)):
        if len(windows) == 1:
            alpha_val = 1.0
        else:
            alpha_val = 0.3 + 0.7 * (i / (len(windows) - 1))
        CND_w = moving_average(CND, window_size=w)
        KPA_w = moving_average(KPA, window_size=w) if KPA is not None else None
        PC_w = moving_average(PC, window_size=w)

        epochs_cnd = list(range(w, w + len(CND_w))) if CND_w is not None else []
        epochs_kpa = list(range(w, w + len(KPA_w))) if KPA_w is not None else []
        epochs_pc = list(range(1 + w, 1 + w + len(PC_w))) if PC_w is not None else []

        if CND_w is not None:
            ax1.plot(epochs_cnd, normalize_data(CND_w),
                    label=f"CND (w={w})", linestyle=":", linewidth=3.5,
                    color=colors["CND"], alpha=alpha_val)
        if KPA_w is not None:
            ax1.plot(epochs_kpa, normalize_data(KPA_w),
                    label=f"KPA (w={w})", linestyle="--", linewidth=3.5,
                    color=colors["KPA"], alpha=alpha_val)
        if PC_w is not None:
            ax1.plot(epochs_pc, normalize_data(PC_w),
                    label=f"PC (w={w})", linestyle="-.", linewidth=3.5,
                    color=colors["PC"], alpha=alpha_val)

    # Axis labels (improved readability and tick parameters)
    ax1.set_ylabel("Normalized Proxy Value", fontsize=22)
    ax2.set_ylabel("Test Accuracy", fontsize=22)
    ax1.set_xlabel("Epoch", fontsize=22)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Title with noise type and intensity if present
    parts = exp_name.split("_")
    if "CIFAR10" in exp_name or "CIFAR100" in exp_name:
        dataset = parts[5]
        noise_type = parts[6]
        noise_level = parts[-1]  # e.g., sim_2
        title_exp = f"{exp_name} ({noise_type}, {noise_level})"
    else:
        title_exp = exp_name
    #ax1.set_title(f"Early Stopping Metrics vs Test Accuracy\n{title_exp}", fontsize=12)

    # Legends for both axes, grouped by metric
    def sort_legend_by_metric(labels, lines):
        groups = defaultdict(list)
        for label, line in zip(labels, lines):
            if "CND" in label:
                groups["CND"].append((label, line))
            elif "KPA" in label:
                groups["KPA"].append((label, line))
            elif "PC" in label:
                groups["PC"].append((label, line))
            else:
                groups["Other"].append((label, line))
        # Group order: CND, KPA, PC, Other
        sorted_items = groups["CND"] + groups["KPA"] + groups["PC"] + groups["Other"]
        if len(sorted_items) == 0:
            return lines, labels
        sorted_labels, sorted_lines = zip(*sorted_items)
        return list(sorted_lines), list(sorted_labels)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    combined_lines = lines1 + lines2
    combined_labels = labels1 + labels2
    sorted_lines, sorted_labels = sort_legend_by_metric(combined_labels, combined_lines)
    ax1.legend(sorted_lines, sorted_labels, fontsize=18, loc="upper right", framealpha=1.0, bbox_to_anchor=(1.0, 1.0))
    ax2.legend(sorted_lines, sorted_labels, fontsize=18, loc="upper right", framealpha=1.0, bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout(pad=2.0)
    # --- Save using exp_name directly as the filename ---
    os.makedirs(f"models/{EXPERIMENT_DIRECTION}/plots", exist_ok=True)
    plt.savefig(f"models/{EXPERIMENT_DIRECTION}/plots/{exp_name}.png")
    plt.close()



def load_experiment_data(base_dir: str) -> dict:
    """
    Loads experiment data from a base directory containing subfolders.
    Each subfolder is expected to have:
      - args.json
      - performances.pkl
    """
    experiment_data = {}
    experiments = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for experiment in experiments:
        experiment_path = os.path.join(base_dir, experiment)
        args_path = os.path.join(experiment_path, "args.json")
        performances_path = os.path.join(experiment_path, "performances.pkl")

        experiment_data[experiment] = {"args": None, "performances": None}

        # Load args.json
        if os.path.exists(args_path):
            with open(args_path, "r") as f:
                try:
                    args_data = json.load(f)
                    experiment_data[experiment]["args"] = args_data
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in {args_path}")

        # Load performances.pkl
        if os.path.exists(performances_path):
            with open(performances_path, "rb") as f:
                try:
                    performances_data = pickle.load(f)
                    experiment_data[experiment]["performances"] = performances_data
                except pickle.UnpicklingError:
                    print(f"Error unpickling file {performances_path}")

    return experiment_data


def filter_experiments(data: dict, filters: set) -> dict:
    """
    Filters a dictionary of experiment data based on a set of (key, value) conditions.
    """
    if not filters:
        return data

    filtered_data = {}
    for experiment, details in data.items():
        if details.get("args"):
            # Check if all filter conditions are satisfied
            include = all(
                details["args"].get(key) == value
                for key, value in filters
            )
            if include:
                filtered_data[experiment] = details
    return filtered_data


# --------------------------------------------------------------------------------
# Main execution wrapped in main()
# --------------------------------------------------------------------------------

def main():
    args_cli = parse_args()
    # Ensure correct working directory
    if "program" in os.getcwd():
        os.chdir("..")
    base_directory = Path("models") / args_cli.experiment_direction

    # Load and filter
    data = load_experiment_data(str(base_directory))
    filters = {('expand_dataset', True)} if args_cli.filter_expand_dataset else set()
    filtered_data = filter_experiments(data, filters) if filters else data

    experiments_to_plot = [exp for exp in args_cli.experiments if exp in filtered_data]
    missing = set(args_cli.experiments) - set(experiments_to_plot)
    if missing:
        print(f"Warning: Experiments not found and will be skipped: {missing}")

    for experiment in experiments_to_plot:
        details = filtered_data[experiment]
        performances = details["performances"]
        # Prepare inputs for plotting
        test_acc = torch.tensor(performances.test_acc, dtype=torch.float32)
        # Compute median CND slice for the last layer
        args_dict = details["args"]
        args = argparse.Namespace(**args_dict)
        if args.dataset == 'NEWS':
            embedding_weights, _, _ = pickle.load(open("data/20news-bydate/news.pkl", "rb"), encoding='iso-8859-1')
        else:
            embedding_weights = None

        _, args = model_definition(device, args, embedding_weights=embedding_weights)
        last_layer = len(args.neurs_x_hid_lyr) - 1

        start_idx = sum(
            neurons for layer_idx, neurons in args.neurs_x_hid_lyr.items()
            if layer_idx < last_layer
        )
        cnd_index = args.CND_type.index(args_cli.cnd_type)
        cnd_slice = performances.CND[cnd_index, :, start_idx:]
        #cnd_median = torch.median(torch.tensor(cnd_slice, dtype=torch.float32), dim=1).values
        cnd_median = torch.quantile(torch.tensor(cnd_slice, dtype=torch.float32), args_cli.cnd_percentile/100, dim=1)

        # Known polluted accuracy and PC tensors
        kpa_tensor = (
            torch.tensor(performances.known_polluted_accuracy, dtype=torch.float32)
            if hasattr(performances, "known_polluted_accuracy")
            else None
        )
        pc_tensor = torch.tensor(performances.PC, dtype=torch.float32)

        # Directly plot and save using new logic
        new_performances_plot(
            test_acc,
            cnd_median,
            kpa_tensor,
            pc_tensor,
            args_cli.windows,
            experiment
        )
        #save_plot_with_path(experiment, args_cli.experiment_direction)


if __name__ == "__main__":
    main()

