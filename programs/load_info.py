import torch
from network_structure import model_definition
import matplotlib.pyplot as plt
import pickle

# ------------------------------------------------------------------------------
# Extract early stopping and memorization metrics for a single experiment
# ------------------------------------------------------------------------------
def extract_experiment_features(experiment, details, windows, cnd_type, patiences, device, cnd_percentile=20):
    """
    Extracts early stopping and memorization metrics for a single experiment.
    Returns a list of result dictionaries.

    :param float cnd_percentile: Percentile (0-100) to use when computing the CND quantile.
    """
    import torch
    from argparse import Namespace
    from scipy.stats import pearsonr

    results = []
    performances = details["performances"]
    args_dict = details["args"]
    if args_dict is None:
        print(f"Skipping {experiment}: args.json is missing or invalid.")
        return results
    args = Namespace(**args_dict)

    if args.dataset == 'NEWS':
        embedding_weights, _, _ = pickle.load(open("data/20news-bydate/news.pkl", "rb"), encoding='iso-8859-1')
    else:
        embedding_weights = None

    _, args = model_definition(device, args, embedding_weights=embedding_weights)
    # Compute label noise ratio
    args.label_noise_ratio = len(args.corrupted_samples) / (
        len(args.not_corrupted_samples) + len(args.corrupted_samples)
    )

    required_attrs = ["CND", "misslab_acc", "PC", "test_acc", "known_polluted_accuracy"]
    if not all(hasattr(performances, attr) for attr in required_attrs):
        print(f"Skipping {experiment}: required attributes missing in performances.")
        return results

    # Convert lists to tensors
    misslab_acc = torch.tensor(performances.misslab_acc, dtype=torch.float32)
    test_acc = torch.tensor(performances.test_acc, dtype=torch.float32)
    pc_values = torch.tensor(performances.PC, dtype=torch.float32)
    known_polluted_accuracy = (
        torch.tensor(performances.known_polluted_accuracy, dtype=torch.float32)
        if hasattr(performances, "known_polluted_accuracy")
        else None
    )


    # Process CND metrics
    idx_cnd_type = args.CND_type.index(cnd_type)

    # Slice last layer
    last_layer = len(args.neurs_x_hid_lyr) - 1
    start_idx = sum(
        neurons for layer_idx, neurons in args.neurs_x_hid_lyr.items()
        if layer_idx < last_layer
    )
    cnd_slice = performances.CND[idx_cnd_type, :, start_idx:]
    #cnd_median = torch.median(torch.tensor(cnd_slice, dtype=torch.float32), dim=1).values
    cnd_median = torch.quantile(torch.tensor(cnd_slice, dtype=torch.float32), cnd_percentile / 100.0, dim=1)
    for w in windows:
        smoothed_cnd = moving_average(cnd_median, window_size=w)
        smoothed_test = moving_average(test_acc, window_size=w) if test_acc is not None else None
        r_val = (
            pearsonr(
                smoothed_cnd.detach().cpu().numpy(),
                smoothed_test.detach().cpu().numpy()
            )[0]
            if smoothed_cnd is not None and smoothed_test is not None
            else None
        )

        for p in patiences:
            best_idx = find_first_maximum(smoothed_cnd, patience=p) if smoothed_cnd is not None else 0
            best_epoch = min(best_idx + w, len(test_acc) - 1)
            memorization_epoch = find_first_minimum(misslab_acc, patience=p)
            best_test_epoch = torch.argmax(test_acc).item()
            delta_test_acc = test_acc[best_test_epoch].item() - test_acc[best_epoch].item()

            results.append({
                "experiment": experiment,
                "seed": args.seed,
                "dataset": args.dataset,
                "lr": args.lr,
                "noise_type": args.noise_type,
                "label_noise_ratio": args.label_noise_ratio,
                "metric": cnd_type,
                "window": w,
                "patience": p,
                "best_epoch": best_epoch,
                "memorization_epoch": memorization_epoch,
                "best_test_epoch": best_test_epoch,
                "delta_test_acc": delta_test_acc,
                "pearson_corr": r_val,
                "test_acc": test_acc[best_test_epoch].item(),
                "ea_test_acc": test_acc[best_epoch].item(),
                "delta_ea_test_acc": test_acc[best_test_epoch].item() - test_acc[best_epoch].item(),
                "delta_best_epoch": best_epoch - best_test_epoch,
            })

    # Process PC metrics
    for w in windows:
        smoothed_pc = moving_average(pc_values, window_size=w)
        smoothed_test = moving_average(test_acc[1:], window_size=w) if test_acc is not None else None #The PC has one element less with respect the other metrics
        r_val_pc = (
            pearsonr(smoothed_pc.detach().cpu().numpy(), smoothed_test.detach().cpu().numpy())[0]
            if smoothed_pc is not None and smoothed_test is not None
            else None
        )
        for p in patiences:
            best_idx_pc = find_first_minimum(smoothed_pc, patience=p) if smoothed_pc is not None else 0
            best_epoch_pc = min(best_idx_pc + w, len(test_acc) - 1)
            memorization_epoch_pc = find_first_minimum(misslab_acc, patience=p)
            best_test_epoch_pc = torch.argmax(test_acc).item()
            delta_test_acc_pc = test_acc[best_test_epoch_pc].item() - test_acc[best_epoch_pc].item()

            results.append({
                "experiment": experiment,
                "seed": args.seed,
                "dataset": args.dataset,
                "lr": args.lr,
                "noise_type": args.noise_type,
                "label_noise_ratio": args.label_noise_ratio,
                "metric": "PC",
                "window": w,
                "patience": p,
                "best_epoch": best_epoch_pc,
                "memorization_epoch": memorization_epoch_pc,
                "best_test_epoch": best_test_epoch_pc,
                "delta_test_acc": delta_test_acc_pc,
                "pearson_corr": r_val_pc,
                "test_acc": test_acc[best_test_epoch_pc].item(),
                "ea_test_acc": test_acc[best_epoch_pc].item(),
                "delta_ea_test_acc": test_acc[best_test_epoch_pc].item() - test_acc[best_epoch_pc].item(),
                "delta_best_epoch": best_epoch_pc - best_test_epoch_pc,
            })

    # Process KPA metrics if available
    if known_polluted_accuracy is not None:
        for w in windows:
            smoothed_kpa = moving_average(known_polluted_accuracy, window_size=w)
            smoothed_test = moving_average(test_acc, window_size=w) if test_acc is not None else None
            r_val_kpa = (
                pearsonr(smoothed_kpa.detach().cpu().numpy(), smoothed_test.detach().cpu().numpy())[0]
                if smoothed_kpa is not None and smoothed_test is not None
                else None
            )
            for p in patiences:
                best_idx_kpa = find_first_minimum(smoothed_kpa, patience=p) if smoothed_kpa is not None else 0
                best_epoch_kpa = min(best_idx_kpa + w, len(test_acc) - 1)
                memorization_epoch_kpa = find_first_minimum(misslab_acc, patience=p)
                best_test_epoch_kpa = torch.argmax(test_acc).item()
                delta_test_acc_kpa = test_acc[best_test_epoch_kpa].item() - test_acc[best_epoch_kpa].item()

                results.append({
                    "experiment": experiment,
                    "seed": args.seed,
                    "dataset": args.dataset,
                    "lr": args.lr,
                    "noise_type": args.noise_type,
                    "label_noise_ratio": args.label_noise_ratio,
                    "metric": "KPA",
                    "window": w,
                    "patience": p,
                    "best_epoch": best_epoch_kpa,
                    "memorization_epoch": memorization_epoch_kpa,
                    "best_test_epoch": best_test_epoch_kpa,
                    "delta_test_acc": delta_test_acc_kpa,
                    "pearson_corr": r_val_kpa,
                    "test_acc": test_acc[best_test_epoch_kpa].item(),
                    "ea_test_acc": test_acc[best_epoch_kpa].item(),
                    "delta_ea_test_acc": test_acc[best_test_epoch_kpa].item() - test_acc[best_epoch_kpa].item(),
                    "delta_best_epoch": best_epoch_kpa - best_test_epoch_kpa,
                })

    return results


def moving_average(data: torch.Tensor, window_size: int = 3) -> torch.Tensor or None:
    """
    Computes the moving average over a 1D tensor with a given window size.
    """
    if len(data) < window_size:
        return None  # Not enough data for a valid moving average
    if window_size <= 1:
        return data.clone()
    
    smoothed = []
    for i in range(window_size-1, len(data)):
        smooth_value = torch.sum(data[i - window_size + 1: i+1]) / window_size
        smoothed.append(smooth_value)

    return torch.tensor(smoothed)


def find_first_maximum(values: torch.Tensor, patience: int = 1) -> int:
    """
    Finds the first local/global maximum index while iterating through `values`,
    explicitly excluding the very first value (index 0) from being considered a maximum.

    Behavior:
    - We only start tracking a candidate maximum after we observe an ascent (values[i] > values[i-1]).
    - `best_value`/`best_index` update on strictly larger values once ascent has begun.
    - Patience counts consecutive non-improvements after the current best.
    - If patience exceeds `patience`, return the last best_index found (or -1 if none).
    - If we reach the end without patience being exceeded, return the best_index (or -1 if none).
    """
    if values is None or len(values) == 0:
        return -1  # or 0 if you prefer, but still not "a max"

    best_value = float("-inf")
    best_index = -1
    current_patience = 0
    has_ascended = False  # ensures index 0 is never treated as a max

    for i in range(1, len(values)):
        if values[i] > values[i-1]:
            has_ascended = True

        if has_ascended:
            if values[i] > best_value:
                best_value = values[i]
                best_index = i
                current_patience = 0
            else:
                current_patience += 1
                if current_patience > patience:
                    return best_index

    return best_index

def find_first_minimum(values: torch.Tensor, patience: int = 1) -> int:
    """
    Finds the first global minimum index while iterating through `values`.
    - `best_value` is the minimum so far.
    - `best_index` is the index of that minimum.
    - Reset patience to 0 when we see a new minimum.
    - If patience exceeds `patience`, return the last best_index found.
    - If we reach the end without exceeding patience, return the best_index.
    """
    # Edge case: if no data or None, return 0 or -1 as you prefer
    if values is None or len(values) == 0:
        return 0

    best_value = values[0]
    best_index = 0
    current_patience = 0

    for i in range(1, len(values)):
        if values[i] < best_value:
            best_value = values[i]
            best_index = i
            current_patience = 0
        else:
            current_patience += 1
            if current_patience > patience:
                return best_index

    return best_index
