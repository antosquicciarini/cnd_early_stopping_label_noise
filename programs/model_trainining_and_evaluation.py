import torch
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from performances import Performances
from critical_sample_ratio import compute_CSR
from prediction_changes import compute_changed_predictions
from graph_entropy import graph_entropy_evaluation
from loss_sesivity import loss_sensivity
from neuron_frequency_activation import neuron_frequency_activation, neuron_frequency_activation_plot
from cnd import cnd, jsd_bins_hist
import numpy as np
from torch.cuda.amp import GradScaler, autocast
import time
from loss_regularizer import apply_regularizers
import logging
import os
import sys
from layerwise_PDF import visualize_layerwise_activation_distributions
import torch.nn.functional as F
from cosine_similarity import gradient_extraction, gradient_extraction_layerwise, compute_gradients_similarity, compute_gradients_similarity_by_layer, plot_cosine_similarities, plot_cosine_similarities_with_rolling_avg, compute_gradients_similarity_fast, compute_gradients_similarity_rand, compute_gradients_dissimilarity_random_classes, compute_gradients_similarity_rand_two_subsets, compute_gradients_similarity_known_corrupted, compute_classwise_gradient_alignment, compute_gradient_alignment_corrupted_reference, compute_updates_similarity, compute_minibatch_gradient_similarity
import matplotlib.pyplot as plt
from cnd_dropout import cnd_dropout_re, cnd_dropout_mask




def model_evaluate(model, loader, device, args, apply_mask=False, apply_mask_only_last_layer=False,
                   stop_criteria_enabled=False, threshold=1e-3, patience=5):
    """
    Evaluate the model with early stopping based on accuracy convergence over recent batches.
    Returns:
    - accuracy: estimated accuracy
    - predictions: tensor of predictions
    """
    model.eval()
    correct = 0
    total = 0
    predictions = torch.empty(0, dtype=torch.long).to(device)
    recent_accuracies = []

    # ▶ smooth metric accumulators
    nll_sum = 0.0
    n_items = 0

    with torch.no_grad():
        for batch_idx, (images, labels, idx) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            output, _ = model(images, idx, apply_mask=apply_mask)

            # ▶ stable log-likelihood
            log_probs = F.log_softmax(output, dim=1)
            # sum NLL over batch for exact mean later
            batch_nll = F.nll_loss(log_probs, labels, reduction='sum')
            nll_sum += batch_nll.item()
            n_items += labels.size(0)

            # accuracy plumbing (unchanged)
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predictions = torch.cat((predictions, predicted))

            # early-stop-on-accuracy convergence (unchanged)
            current_accuracy = correct / total if total > 0 else 0
            if stop_criteria_enabled:
                recent_accuracies.append(current_accuracy)
                if len(recent_accuracies) > patience:
                    recent_accuracies.pop(0)
                if len(recent_accuracies) == patience:
                    max_diff = max(recent_accuracies) - min(recent_accuracies)
                    if max_diff < threshold:
                        break

    accuracy = correct / total if total > 0 else 0

    # ▶ stash smooth metrics for the caller (don’t change return signature)
    if n_items > 0:
        avg_nll = nll_sum / n_items
        avg_logp = -avg_nll
    else:
        avg_nll = float("nan")
        avg_logp = float("nan")
    args._last_eval_nll = avg_nll
    args._last_eval_avg_logp = avg_logp

    return accuracy, predictions



# def model_evaluate_Maini(model, loaders['train_loader_corrupted'], loaders['train_loader_not_corrupted'], device, args, apply_mask=False):

#     print("rest mode results")
#     model.set_drop_mode("test")
#     accuracy_corrupted, _ = model_evaluate(model, loaders['train_loader_corrupted'], device, args)
#     accuracy_non_corrupted, _ = model_evaluate(model, loaders['train_loader_not_corrupted'], device, args)
#     print(f"Accuracy corrupted samples: {accuracy_corrupted * 100:.3f}%")
#     print(f"Accuracy non-corrupted samples: {accuracy_non_corrupted * 100:.3f}%")

#     print("Drop mode results")
#     model.set_drop_mode("drop")
#     accuracy_corrupted, _ = model_evaluate(model, loaders['train_loader_corrupted'], device, args)
#     accuracy_non_corrupted, _ = model_evaluate(model, loaders['train_loader_not_corrupted'], device, args)
#     print(f"Accuracy corrupted samples: {accuracy_corrupted * 100:.3f}%")
#     print(f"Accuracy non-corrupted samples: {accuracy_non_corrupted * 100:.3f}%")

#     args.accuracy_corrupted = accuracy_corrupted
#     args.accuracy_non_corrupted = accuracy_non_corrupted
#     return args


def train_and_evaluate_model(
    model, loaders, criterion, optimizer, scheduler, device, args, logger):

    epochs = args.epochs

    prev_predictions = None  # To store predictions from the previous epoch
    model_mask_cnd_dr = None
    grad_past = None
    grad = None

    performances = Performances()
    frequency_activation_list = []
    cos_sim_list = []
    cos_sim_rand_list = []
    cos_sim_known_list = []

    # Early stopping variables
    if getattr(args, "early_stopping", False):  # Check if early stopping should be activated
        patience = getattr(args, "early_stopping_patience", 10)  # Default patience
        best_test_accuracy = 0
        epochs_without_improvement = 0

    # Enable mixed precision and other optimizations if CUDA is detected
    use_amp = args.code_cloud and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None

    print(model)
    print(sum(p.numel() for p in model.parameters()))

    # Prepare to track epoch-wise parameter updates

    for epoch in range(epochs):
        # Track gradient alignment per iteration for this epoch
        GA_list = []

        # Track the start time of the epoch
        start_time = time.time()

        # Training phase
        args.current_epoch = epoch
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        performances_dict = {}
        epoch_por = (epoch+1)/ epochs
        model_mask_cnd_dr = None

        logger.info(f"Epoch {epoch + 1}/{epochs} ..")

        # Store predictions for this epoch
        current_predictions = torch.empty(0, dtype=torch.long).to(device)
        if not args.code_cloud:
            from tqdm import tqdm
            loaders['train_loader'] = tqdm(loaders['train_loader'], desc=f"Epoch {epoch}", unit="batch", leave=True)

        for ii, (images, labels, indx) in enumerate(loaders['train_loader']):

            if model_mask_cnd_dr is not None:
                model.set_mask(model_mask_cnd_dr)

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            indx = indx.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Mixed precision training
            if use_amp:
                with autocast():
                    outputs, pre_activations = model(images, indx, return_intermediates=True)
                    loss = criterion(outputs, labels)
                    loss = apply_regularizers(loss, model, labels, pre_activations, epoch_por, args)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs, pre_activations = model(images, indx, return_intermediates=True)  # Forward pass
                loss = criterion(outputs, labels)  # Calculate loss
                loss = apply_regularizers(loss, model, labels, pre_activations, epoch_por, args)

                loss.backward()  # Backward pass
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()  # Update weights

            # Update grad_past using exponential decay if batch_acc is enabled
            if getattr(args, "GA_momentum", False):
                if grad_past is None:
                    grad_past = grad
                else:
                    grad_past = [args.GA_momentum_decay_factor * gp + (1 - args.GA_momentum_decay_factor) * g for gp, g in zip(grad_past, grad)]
            else:
                grad_past = grad

            if getattr(args, "GA", False) or getattr(args, "GA_momentum", False):
                grad = gradient_extraction_layerwise(model)
                if grad_past is not None:
                    GA = compute_updates_similarity(grad, grad_past)
                    GA_list.append(GA)
                    if (getattr(args, "GA_filter", None) == "up" and GA > 0) or (getattr(args, "GA_filter", None) == "down" and GA < 0):
                        continue
                    
            elif getattr(args, "GA_half", False):
                GA = compute_minibatch_gradient_similarity(model, criterion, images, labels, device, args)
                GA_list.append(GA)

            # Calculate training accuracy
            _, predicted_train = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

            if getattr(args, "debug_reduced_epoch_duration", None):  # Default to None if not present
                if total_train > args.debug_reduced_epoch_duration:
                    break

            if getattr(args, "CGA", False):
                cos_sim_list.append(compute_gradients_dissimilarity_random_classes(model, criterion, images, labels, indx, device, args))
            if hasattr(args, "CGA_random"):
                GA_rand = compute_gradients_similarity_rand_two_subsets(model, criterion, images, labels, indx, device, args)
                cos_sim_rand_list.append(GA_rand)

            if getattr(args, "CGA_known", False):
                grad_subset_or = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None]).detach()
                cos_sim_known_list.append(compute_gradients_similarity_known_corrupted(model, grad_subset_or, criterion, loaders['train_loader_known_corrupted'], device, args))
            if getattr(args, 'cnd_dropout_mask', False):
                model.disable_mask()
                _, pre_activations = model(images, indx, return_intermediates=True)
                cnd_result = jsd_bins_hist(pre_activations.detach().cpu(), labels.cpu(), args)
                model_mask_cnd_dr = cnd_dropout_mask(cnd_result, getattr(args, "cnd_dropout_treshold", 0.25), device, args)
                model.set_mask(model_mask_cnd_dr)

            running_loss += loss.item()  # Accumulate loss
            grad_past = grad

        # Disable ALL the masks for the evaluation
        model.disable_mask()

        # Update learning rate
        scheduler.step() if scheduler is not None else None
       
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch + 1}/{epochs}, Current LR: {current_lr:.6f}")

        # Track and print the end time of the epoch
        end_time = time.time()
        epoch_duration = end_time - start_time
        logger.info(f"Epoch {epoch + 1}/{epochs} training completed in {epoch_duration:.2f} seconds.")

        train_loss = running_loss / ii if ii != 0 else running_loss
        train_accuracy = correct_train / total_train if total_train != 0 else 0
        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(loaders['train_loader']):.4f}, Train Accuracy: {train_accuracy * 100:.2f}%")


        # Evaluation phase
        if epoch == epochs-1:
            stop_criteria_enabled = False
        else:
            stop_criteria_enabled = True

        test_accuracy, _ = model_evaluate(model, loaders['test_loader'], device, args, stop_criteria_enabled=stop_criteria_enabled)
        logger.info(f"Epoch {epoch + 1}/{epochs}, Test Accuracy: {test_accuracy * 100:.2f}%")


        # Early stopping variables
        if getattr(args, "early_stopping", False) and args.current_epoch == 0:
            patience = getattr(args, "early_stopping_patience", 10)  # Default patience for early stopping
            best_test_accuracy = 0
            epochs_without_improvement = 0  # Counter for early stopping
            lr_plato_counter = 0            # Separate counter for lr_plato reduction

        # Inside the training loop after evaluation:
        if getattr(args, "early_stopping", False):
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                epochs_without_improvement = 0  # Reset early stopping counter
                lr_plato_counter = 0            # Reset lr_plato counter as well
                logger.info("Early stopping: new best accuracy.")
            else:
                epochs_without_improvement += 1
                lr_plato_counter += 1
                remaining_epochs = patience - epochs_without_improvement
                logger.info(f"Early stopping: {remaining_epochs} epoch(s) left before patience limit is reached.")

                # LR reduction logic for "lr_plato" policy using a separate counter
                if getattr(args, "lr_policy", False) == "lr_plato":
                    if lr_plato_counter >= args.lr_plato_patience:
                        current_lr = optimizer.param_groups[0]['lr']
                        new_lr = current_lr * args.lr_gamma
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = new_lr
                        logger.info(f"LR Policy 'lr_plato': No improvement for {args.lr_plato_patience} epochs. "
                                    f"Reducing LR from {current_lr:.6f} to {new_lr:.6f}.")
                        lr_plato_counter = 0  # Reset lr_plato counter

            if epochs_without_improvement >= patience:
                logger.warning(f"Early stopping activated at epoch {epoch + 1}. No improvement for {patience} epochs.")
                break

        # Track and print the end time of the epoch
        end_time = time.time()
        epoch_duration = end_time - start_time
        logger.info(f"Epoch {epoch + 1}/{epochs} training and evaluation completed in {epoch_duration:.2f} seconds.")


        # Check if accuracy is below the threshold
        accuracy_threshold = (1 + getattr(args, "epsilon", 0.5)) / args.num_classes
        if test_accuracy < accuracy_threshold and epoch <= 3 and getattr(args, "fast_training", False):
            logger.warning(f"\nTerminating training: Test accuracy {test_accuracy:.4f} is below the threshold {accuracy_threshold:.4f}.\n")
            args.broken_training = True
            break
        else:
            args.broken_training = False


        # EVALUATE STEP
        if loaders['train_loader_corrupted'] is not None:
            if "KPA" in args.metrics:

                polluted_accuracy, _ = model_evaluate(model, loaders['train_loader_corrupted'], device, args, stop_criteria_enabled=stop_criteria_enabled)

                logger.info(f"Corrupted Sample Accuracy {epoch + 1}/{epochs}: {polluted_accuracy * 100:.2f}% ")
                
                if hasattr(args, "noise_type") and  args.noise_type != "clean_label":
                    expected_accuracy = 100 * (1 - test_accuracy) / 9
                    logger.info(f"Expected {expected_accuracy:.2f}%")
                elif args.noise_type == "hard_noise":
                    expected_accuracy = 100 / args.num_classes
                    logger.info(f"Expected {expected_accuracy:.2f}%")

            else:
                polluted_accuracy = None

            if "KPA_loglik" in args.metrics:
                # still call model_evaluate so args._last_eval_* get updated
                _, _ = model_evaluate(
                    model, loaders['train_loader_corrupted'], device, args,
                    stop_criteria_enabled=stop_criteria_enabled
                )

                # Read smooth stats stashed by model_evaluate
                avg_logp = getattr(args, "_last_eval_avg_logp", float("nan"))
                avg_nll  = getattr(args, "_last_eval_nll",    float("nan"))

                # Save what you want in known_polluted_accuracy_loglik instead of polluted_accuracy
                known_polluted_accuracy_loglik = avg_logp

                # A simple random-guess baseline for comparison
                C = getattr(args, "num_classes", None) or loaders['train_loader_corrupted'].dataset.num_classes
                baseline_logp = -np.log(C)

                logger.info(
                    f"Corrupted samples — Avg log-likelihood: {avg_logp:.4f} | "
                    f"Avg NLL: {avg_nll:.4f} | "
                    f"Uniform-baseline log p: {baseline_logp:.4f}"
                )

                # keep your 'expected' computation if needed
                if hasattr(args, "noise_type") and args.noise_type != "clean_label":
                    expected_accuracy = 100 * (1 - test_accuracy) / 9
                    logger.info(f"Expected {expected_accuracy:.2f}%")
                elif getattr(args, "noise_type", None) == "hard_noise":
                    expected_accuracy = 100 / args.num_classes
                    logger.info(f"Expected {expected_accuracy:.2f}%")
                            
            else:
                known_polluted_accuracy_loglik = None


        if "train_loader_known_corrupted" in loaders:
            
            if "KPA" in args.metrics:
                known_polluted_accuracy, _ = model_evaluate(model, loaders['train_loader_known_corrupted'], device, args)
                logger.info(f"Known Corrupted Sample Accuracy {epoch + 1}/{epochs}: {known_polluted_accuracy * 100:.2f}% ")
            else:
                known_polluted_accuracy = None

            if "KPA_loglik" in args.metrics:
                # still call model_evaluate so args._last_eval_* get updated
                _, _ = model_evaluate(
                    model, loaders['train_loader_known_corrupted'], device, args,
                    stop_criteria_enabled=stop_criteria_enabled
                )

                # Read smooth stats stashed by model_evaluate
                avg_logp = getattr(args, "_last_eval_avg_logp", float("nan"))
                avg_nll  = getattr(args, "_last_eval_nll",    float("nan"))

                # Save what you want in known_polluted_accuracy_loglik instead of polluted_accuracy
                known_polluted_accuracy_loglik = avg_logp

                # A simple random-guess baseline for comparison
                C = getattr(args, "num_classes", None) or loaders['train_loader_corrupted'].dataset.num_classes
                baseline_logp = -np.log(C)

                logger.info(
                    f"Corrupted samples — Avg log-likelihood: {avg_logp:.4f} | "
                    f"Avg NLL: {avg_nll:.4f} | "
                    f"Uniform-baseline log p: {baseline_logp:.4f}"
                )

                # keep your 'expected' computation if needed
                if hasattr(args, "noise_type") and args.noise_type != "clean_label":
                    expected_accuracy = 100 * (1 - test_accuracy) / 9
                    logger.info(f"Expected {expected_accuracy:.2f}%")
                elif getattr(args, "noise_type", None) == "hard_noise":
                    expected_accuracy = 100 / args.num_classes
                    logger.info(f"Expected {expected_accuracy:.2f}%")
            else:
                known_polluted_accuracy_loglik = None     
    

        if loaders['train_loader_fixed'] is not None:

            if "CND" in args.metrics:
                performances_dict, current_predictions = cnd(loaders['train_loader_fixed'], model, device, performances_dict, "CND", args, logger)
                # Create cnd_dropout function
                if getattr(args, 'cnd_dropout_re', False) and args.current_epoch%args.cnd_dropout_patience == 0 and args.current_epoch!=0:
                    # Here we reinitialize the neurons with low CND
                    model = cnd_dropout_re(model, performances_dict['CND_PMF'],  getattr(args, "cnd_dropout_treshold", 0.25), args)
            else:
                current_predictions = None

            if "CND_noisy" in args.metrics and hasattr(args, "noise_type") and args.noise_type != "clean_label":
                performances_dict = cnd(loaders['train_loader_corrupted'], model, device, performances_dict, "CND_noisy", args, logger)

            if "FA" in args.metrics:
                FA_per_neuron, FA_per_instance = neuron_frequency_activation(
                    loaders['train_loader_fixed'], loaders['train_loader_corrupted'], loaders['train_loader_not_corrupted'], model, args.corrupted_samples, device, args
                )

                mask = ((FA_per_neuron > torch.quantile(FA_per_neuron, 0.1))).float()
                model.mask = mask

                polluted_accuracy_mask, _ = model_evaluate(model, loaders['train_loader_corrupted'], device, args, apply_mask=True)
                logger.info(f"Epoch {epoch + 1}/{epochs}, Polluted Sample Accuracy with neuron mask: {polluted_accuracy_mask * 100:.2f}%")
                test_accuracy_mask, _ = model_evaluate(model, loaders['test_loader'], device, args, apply_mask=True)
                logger.info(f"Epoch {epoch + 1}/{epochs}, Test Accuracy with neuron mask: {test_accuracy_mask * 100:.2f}%")
                performances_dict['train_accuracy_mask'] = polluted_accuracy_mask
                performances_dict['test_accuracy_mask'] = test_accuracy_mask

            if "PC" in args.metrics:
                if current_predictions is None:
                    _, current_predictions = model_evaluate(model, loaders['train_loader_not_corrupted'], device, args)

                if prev_predictions is not None:
                    PC = compute_changed_predictions(prev_predictions, current_predictions)
                    logger.info(f"Epoch {epoch + 1}/{epochs}, Number of samples with changed predictions: {PC}")
                    performances_dict['PC'] = PC
                    
                prev_predictions = current_predictions.clone()  # Update previous predictions

            if "LS" in args.metrics:
                LS = loss_sensivity(loaders['train_loader_fixed'], model, criterion, device)
                logger.info(f"Epoch {epoch + 1}/{epochs}, LS: {LS}")
                performances_dict['LS'] = LS

            if "CSR" in args.metrics:
                CSR = compute_CSR(loaders['train_loader_fixed'], model, criterion, device)
                logger.info(f"Epoch {epoch + 1}/{epochs}, CSR: {CSR}")
                performances_dict['CSR'] = CSR

            if "CGE" in args.metrics:
                CGE = graph_entropy_evaluation(loaders['train_loader_fixed'], model, device)
                logger.info(f"Epoch {epoch + 1}/{epochs}, Average CGE: {np.mean(CGE)}")
                performances_dict['CGE'] = CGE

            if getattr(args, "CGA_data_filter_epoch", -1) == args.current_epoch:
                #compute_gradient_alignment_corrupted_reference(model, loaders, criterion, device, args, logger=logger)
                compute_classwise_gradient_alignment(model, loaders, criterion, device, args, logger=logger)

        if getattr(args, "plot_layerwise_act", False):
            visualize_layerwise_activation_distributions(loaders['train_loader_not_corrupted'], loaders['train_loader_corrupted'], model, device, args)

        end_time = time.time()
        epoch_duration = end_time - start_time
        logger.info(f"Epoch {epoch + 1}/{epochs} training, evaluation, and memorization metrics completed in {epoch_duration:.2f} seconds.")

        # Save per-iteration GA values into performances_dict
        performances_dict["GA_list"] = GA_list
        # Update performances
        performances.update(train_loss, train_accuracy, test_accuracy, polluted_accuracy, known_polluted_accuracy_loglik, known_polluted_accuracy, performances_dict, args)




    # if "Maini" in args.network:
    #     args = model_evaluate_Maini(model, loaders['train_loader_corrupted'], loaders['train_loader_not_corrupted'], device, args)

    if getattr(args, "CGA", False):
        plot_cosine_similarities_with_rolling_avg(cos_sim_list, args)
    if hasattr(args, "CGA_random"):
        plot_cosine_similarities_with_rolling_avg(cos_sim_rand_list, args, lab="CGA_random")
    if getattr(args, "CGA_known", False):
        plot_cosine_similarities_with_rolling_avg(cos_sim_known_list, args, lab="CGA_known")
        
    for ga_type in ["GA", "GA_half", "GA_momentum"]:
        if getattr(args, ga_type, False):
            plot_cosine_similarities_with_rolling_avg(performances.GA, args, lab=ga_type)
            performances.get_flattened_GA()


    return model, performances, args






# PLOT TINY-IMAGENET IMAGES
# import matplotlib.pyplot as plt
# import torchvision
# import torch

# # De-normalize images
# mean = torch.tensor([0.4802, 0.4481, 0.3975]).view(1, 3, 1, 1)
# std = torch.tensor([0.2302, 0.2265, 0.2262]).view(1, 3, 1, 1)
# images_denorm = images[:16] * std + mean
# images_denorm = torch.clamp(images_denorm, 0, 1)

# # Plot images with human-readable class names
# fig, axes = plt.subplots(4, 4, figsize=(10, 10))
# for i, ax in enumerate(axes.flat):
#     img = images_denorm[i].permute(1, 2, 0).cpu()
#     label_idx = labels[i].item()
#     class_name = args.idx_to_name[label_idx]  # ← human-readable name
#     ax.imshow(img)
#     ax.set_title(class_name, fontsize=8)
#     ax.axis('off')

# plt.tight_layout()
# plt.show()