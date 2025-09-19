import os
import sys
import json
import pickle
import itertools
import random
import logging
import multiprocessing
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

# Import your project-specific modules
from network_structure import model_definition
from model_trainining_and_evaluation import train_and_evaluate_model
from load_dataset import load_dataset
from parameter_settings import parameter_settings
from loss_functions import define_loss_function


def generate_settings_combinations(original_dict):
    """
    Generate all possible combinations of parameter settings.
    """
    list_keys = [key for key, value in original_dict.items() if isinstance(value, list)]
    combinations = list(itertools.product(*[original_dict[key] for key in list_keys]))
    result = []

    for combo in combinations:
        new_dict = original_dict.copy()
        for key, value in zip(list_keys, combo):
            new_dict[key] = value
        result.append(new_dict)

    return result


def run_simulation(params, code_cloud, job_name, sim_index=0):
    """
    Execute a single simulation based on the provided parameters.
    """
    # Device setup moved inside the function to avoid global CUDA initialization
    import sys
    import torch

    # Prepare arguments and logging
    args = parameter_settings(params, job_name, sim_index)
    args.code_cloud = code_cloud

    # Fix the random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set up logger for this simulation
    logger_name = f"{args.filename}_{sim_index}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Remove any existing handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create file handler which logs messages
    fh = logging.FileHandler(os.path.join(args.results_dir, f'{args.filename}.log'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Add a console handler only if args.code_cloud is False
    if not args.code_cloud:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # Print the arguments at the beginning of the simulation
    logger.info(f"Starting simulation with arguments:\n{args}")

    # Device and data loader setup
    if args.code_cloud and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        args.n_GPUs = torch.cuda.device_count()
        logger.info(f"CUDA detected. Number of devices: {args.n_GPUs}")
        for i in range(args.n_GPUs):
            logger.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
        num_workers = 0 * args.n_GPUs
        if getattr(args, 'multiple_sim_on_GPU', False):
            num_workers = 0  # By setting num_workers=0, data is loaded in the main process
        args.batch_size *= args.n_GPUs
        logger.info(f"Number of workers: {num_workers}")
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        logger.info(f"Using device: {device}")
        num_workers = 0
        logger.info(f"Number of workers: {num_workers}")

    # Data preparation
    train_dataset, test_dataset, embedding_weights, args = load_dataset(args)
    pin_memory = args.code_cloud

    # Create loaders
    loaders = {
        "train_loader": DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory
        ),
        "train_loader_fixed": DataLoader(
            train_dataset, batch_size=args.fixed_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        ),
        "test_loader": DataLoader(
            test_dataset, batch_size=args.fixed_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        ),
        "train_loader_corrupted": DataLoader(
            Subset(train_dataset, args.corrupted_samples), batch_size=args.fixed_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        ),
        "train_loader_not_corrupted": DataLoader(
            Subset(train_dataset, args.not_corrupted_samples), batch_size=args.fixed_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )
    }

    # Add train_loader_known_corrupted if there are new_indexes
    if len(train_dataset.new_indexes) > 0:
        loaders["train_loader_known_corrupted"] = DataLoader(
            Subset(train_dataset, train_dataset.new_indexes), batch_size=args.fixed_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )

    # Model definition
    model, args = model_definition(device, args, embedding_weights=embedding_weights)

    criterion = define_loss_function(args)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov
    )

    if getattr(args, "lr_policy", None) == "lr_multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma
        )
    elif getattr(args, "lr_policy", None) == "cosine_decay":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=0  # You can adjust eta_min if needed
        )
    else:
        scheduler = None  # No scheduler for constant learning rate

    # Save simulation arguments
    with open(os.path.join(args.results_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Train and evaluate
    model, performances, args = train_and_evaluate_model(
        model, loaders, criterion, optimizer, scheduler, device, args, logger
    )

    # Save results
    if not args.broken_training:
        performances.torch_transformation()
        performances.plot_performances(args)
        with open(os.path.join(args.results_dir, "performances.pkl"), "wb") as f:
            pickle.dump(performances, f)

    # Save the trained model
    model_path = os.path.join(args.results_dir, f"{args.filename}_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args)  # Save args for reproducibility
    }, model_path)
    logger.info(f"Model saved at: {model_path}")

    logger.info(f"SIMULATION COMPLETED: {args.filename}")

    # Clean up handlers to prevent issues in future simulations
    logger.removeHandler(fh)
    if not args.code_cloud:
        logger.removeHandler(ch)

    logger.info(f"SIMULATION COMPLETED: {args.filename}")


def main(job_name, code_cloud):
    """
    Entry point for running simulations.
    """
    # Load job configuration
    file_name = f"programs/simulations/{job_name}.json"
    with open(file_name, 'r') as f:
        json_dict = json.load(f)

    multiple_sim_on_GPU = json_dict.get('multiple_sim_on_GPU', False)
    n_sims = json_dict.get('n_sims', 1)
    json_dict_comb_list = generate_settings_combinations(json_dict)

    print(f"Total simulations to run: {len(json_dict_comb_list)}")

    # Run simulations either in parallel or sequentially
    if multiple_sim_on_GPU and n_sims > 1:
        print("Running simulations in parallel...")
        # Use 'spawn' start method for multiprocessing
        pool = Pool(processes=n_sims)
        args_list = [(params, code_cloud, job_name, i) for i, params in enumerate(json_dict_comb_list)]
        pool.starmap(run_simulation, args_list)
        pool.close()
        pool.join()
    else:
        print("Running simulations sequentially...")
        for i, params in enumerate(json_dict_comb_list):
            print(f"\nSimulation {i+1}/{len(json_dict_comb_list)}\n")
            run_simulation(params, code_cloud, job_name, sim_index=i)


if __name__ == '__main__':
    # Set multiprocessing start method to 'spawn'
    multiprocessing.set_start_method("spawn", force=True)

    # Adjust working directory
    if "program" in os.getcwd():
        os.chdir("..")

    print("Start!")
    if len(sys.argv) > 1:
        code_cloud = True
        job_name_list = [sys.argv[2]]
    else:
        code_cloud = False
        # Import caffeine only if not in cloud execution
        import caffeine
        caffeine.on(display=False)
        print("No CLOUD execution.")
        job_name_list = ['early_stopping_CIFAR10_real_noise_preact_KPA_log']  # [early_stopping_CIFAR10_real_noise_preact_KPA_log  NEWS_CND_dropout CND_tinyimagenet NEWS_CND_yu19_main_exp NEWS_CND_Yu19_network NEWS_CND CND_tinyimagenet CIFAR100_class_gradient_agreement MNIST_class_gradient_agreement CIFAR10_class_gradient_agreement early_stopping_CIFAR10_none MNIST_class_gradient_agreement, MNIST_FashMNIST_cnd_dropout_re  dropout_CND_mask_MNIST_FashMNIST MNIST_class_gradient_agreement early_stopping_CIFAR10_real_noise_preact early_stopping_CIFAR10_real_noise CIFAR100_real_noise_resnet9 SVHN_LeNet CIFAR10_real_noise_resnet9 MNIST_FashMNIST_baseline early_stopping_tinyimagenet MNIST_FashMNIST_cnd_dropout_debug MNIST_FashMNIST_cnd_dropout MNIST_FashMNIST_reg_baseline early_stopping_tinyimagenet early_stopping_CIFAR100_real_noise   early_stopping_CIFAR10_real_noise. early_stopping_CIFAR100_real_noise_err early_stopping_CIFAR10_real_noise_lr_big early_stopping_CIFAR10_real_noise - dataset_noisy_expansion_CIFAR10_gaussian_and_LN - CGA_CIFAR10_symmetric_detect_LN_samples CGA_MNIST_symmetric_detect_LN_samples early_stopping_CIFAR10_real_noise MNIST_CGA_memorised_subset 'early_stopping_CIFAR10_real_noise'] MNIST_class_gradient_agreement # CIFAR10_class_gradient_agreement MNIST_baseline_models CIFAR10_baseline CIFAR100_baseline MNIST_class_gradient_agreement ['CIFAR10_label_perturbation']#['CIFAR10_class_gradient_agreement'] #["CIFAR10_label_perturbation"]#["MNIST_over_LN_data", "CIFAR10_CND_over_missalb"] #["CIFAR10_baseline"] #"MNIST_FashMNIST_label_perturbation""CIFAR10_CND_over_missalb" "MNIST_baseline_models"#"MNIST_FashMNIST_label_perturbation"#"MNIST_FashMNIST_LN_reg"#"MNIST_FashMNIST_baseline" #"CIFAR10_CND_over_missalb" #"MNIST_FashMNIST_over_misslab" #"CIFAR10N_recreate_plot" #"CIFAR10_baseline" #"SVHN_baseline" #"MNIST_FashMNIST_reg" #"basic_dataset_KDE_FFT_incredible_model" #"MNIST_FashMNIST_reg" #"SVHN_LeNet"#"SVHN_LeNet" #"MNIST_FashMNIST_baseline" #"SVHN_LeNet" #"SVHN_LeNet"#"SVHN_CIFAR10_reg_baseline" # "SVHN_CIFAR10_reg" #"basic_dataset_KDE_FFT_test" #"MNIST_FashMNIST_reg_baseline" # "MNIST_FashMNIST_reg" #"basic_dataset_jsd_reg" #"CIFAR10_jsd_reg_adjusted" #"basic_dataset_jsd_reg" #"SVHN_jsd_reg" #"basic_dataset_save_a_model"#"SVHN_jsd_reg_reference" #"basic_dataset_jsd_reg"
        #MNIST_FashMNIST_LN_reg

    for job_name in job_name_list:
        main(job_name, code_cloud)