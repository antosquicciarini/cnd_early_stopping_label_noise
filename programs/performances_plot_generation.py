#!/usr/bin/env python3
import os
import json
import pickle
import argparse
from types import SimpleNamespace

def load_performances(performances_path):
    """Load the pickled Performances object."""
    with open(performances_path, 'rb') as f:
        return pickle.load(f)

def load_args(args_path):
    """Load training args from JSON and convert to a simple namespace."""
    with open(args_path, 'r') as f:
        args_dict = json.load(f)
    return SimpleNamespace(**args_dict)

def main(results_dir):
    # Paths
    perf_path = os.path.join(results_dir, 'performances.pkl')
    args_path = os.path.join(results_dir, 'args.json')

    # Sanity checks
    if not os.path.isfile(perf_path):
        raise FileNotFoundError(f"No performances.pkl found at {perf_path}")
    if not os.path.isfile(args_path):
        raise FileNotFoundError(f"No args.json found at {args_path}")

    # Load
    print(f"Loading performances from {perf_path}…")
    performances = load_performances(perf_path)

    print(f"Loading args from {args_path}…")
    args = load_args(args_path)

    # Ensure the results_dir attribute is set correctly
    args.results_dir = results_dir

    # (Optional) if your plotting methods need timestamp, 
    # make sure args.timestamp exists or extract it from the directory name.
    if not hasattr(args, 'timestamp'):
        args.timestamp = os.path.basename(results_dir)

    # Transform and plot
    print("Transforming performances for plotting…")
    #performances.torch_transformation()

    print("Generating all plots…")
    performances.plot_performances(args)

    print("Done! Plots saved under:", results_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Load a Performances object and regenerate its diagnostic plots.'
    )
    parser.add_argument(
        'results_dir',
        nargs='?',
        default='models/early_stopping_variab_CIFAR10_FAST/exp_2025_03_24_19_04_19_CIFAR10_real_noise_resnet9_sim_0',
        help='Path to the folder containing performances.pkl and args.json (default: %(default)s)'
    )
    args = parser.parse_args()
    main(args.results_dir)