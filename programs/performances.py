import matplotlib.pyplot as plt
import torch
from datetime import datetime
import os
import numpy as np
import pandas as pd
from scipy.signal import hilbert, correlate
from scipy.stats import kendalltau
from sklearn.metrics import mutual_info_score
from matplotlib.colors import to_rgba

def masked_std(tensor, axis=1):
    """
    Calculate the standard deviation of a tensor along a specified axis, ignoring 0 values.

    Parameters:
    - tensor (torch.Tensor): The input tensor.
    - start_idx (int): Start index for slicing the tensor along columns.
    - end_idx (int): End index for slicing the tensor along columns.
    - axis (int): The axis along which to calculate the standard deviation (default is 1).

    Returns:
    - torch.Tensor: Standard deviation values calculated along the specified axis, ignoring 0s.
    """
    mask = tensor != 0  # Mask to ignore 0 values

    # Mean ignoring zeros
    mean_values = (tensor * mask).sum(dim=axis) / mask.sum(dim=axis).clamp(min=1)
    
    # Variance ignoring zeros
    variance = ((tensor - mean_values.unsqueeze(axis)) ** 2 * mask).sum(dim=axis) / mask.sum(dim=axis).clamp(min=1)
    
    # Standard deviation
    std_values = torch.sqrt(variance)
    return std_values


def masked_mean(tensor, axis=1):
    """
    Calculate the mean of a tensor along a specified axis, ignoring 0 values.

    Parameters:
    - tensor (torch.Tensor): The input tensor.
    - start_idx (int): Start index for slicing the tensor along columns.
    - end_idx (int): End index for slicing the tensor along columns.
    - axis (int): The axis along which to calculate the mean (default is 1).

    Returns:
    - torch.Tensor: Mean values calculated along the specified axis, ignoring 0s.
    """
    mask = tensor != 0  # Create a mask for non-zero values

    # Compute mean where mask is True
    mean_values = (tensor * mask).sum(dim=axis) / mask.sum(dim=axis).clamp(min=1)
    return mean_values


def is_all_none(lst):
    return all(element is None for element in lst)

def n_neurs_l(l_idx,  sample_dict):
    return sum([value for key, value in sample_dict.items() if key<l_idx])

def compute_correlation_metrics(vect_1, vect_2, apply_lag=False):
    # Ensure both tensors are numpy arrays
    vect_1 = np.asarray(vect_1)
    vect_2 = np.asarray(vect_2)

    # Adjust the size of the vectors if they are different
    min_len = min(len(vect_1), len(vect_2))
    if len(vect_1) > min_len:
        vect_1 = vect_1[-min_len:]  # Cut from the beginning of the longer vector
    if len(vect_2) > min_len:
        vect_2 = vect_2[-min_len:]

    metrics = {}

    # 1. Cross-correlation function to find the best phase shift
    def cross_correlation(x, y):
        cross_corr = correlate(x, y, mode='full')
        lags = np.arange(-len(x) + 1, len(x))
        max_corr = np.max(cross_corr)
        best_lag = lags[np.argmax(cross_corr)]
        return max_corr, best_lag

    # 2. Phase Locking Value (PLV)
    def phase_locking_value(x, y):
        # Compute the Hilbert transform to get the analytic signal
        analytic_x = hilbert(x)
        analytic_y = hilbert(y)
        phase_x = np.angle(analytic_x)
        phase_y = np.angle(analytic_y)
        
        # Phase difference
        phase_diff = phase_x - phase_y
        plv = np.abs(np.sum(np.exp(1j * phase_diff)) / len(phase_diff))
        return plv

    # 3. Time-lagged Mutual Information
    def time_lagged_mutual_information(x, y, lag=0):
        if lag > 0:
            x_shifted = x[:-lag]
            y_shifted = y[lag:]
        elif lag < 0:
            x_shifted = x[-lag:]
            y_shifted = y[:lag]
        else:
            x_shifted = x
            y_shifted = y

        # Discretize the continuous values (necessary for mutual information)
        x_binned = np.digitize(x_shifted, bins=np.histogram_bin_edges(x_shifted, bins='auto'))
        y_binned = np.digitize(y_shifted, bins=np.histogram_bin_edges(y_shifted, bins='auto'))

        mi = mutual_info_score(x_binned, y_binned)
        return mi

    # 4. Instantaneous Phase Difference using Hilbert transform
    def instantaneous_phase_difference(x, y):
        analytic_x = hilbert(x)
        analytic_y = hilbert(y)
        phase_x = np.angle(analytic_x)
        phase_y = np.angle(analytic_y)
        phase_diff = phase_x - phase_y
        return phase_diff

    # 5. Kendall's Tau
    def kendall_tau_with_lags(x, y, max_lag=5, apply_lag=False):
        best_tau = None
        best_p_value = None
        best_lag = None

        # If not applying lags, use only the original sequences
        if not apply_lag:
            tau, p_value = kendalltau(x, y)
            return tau, p_value, 0  # No lag applied

        # Loop through lags if applying lag is enabled
        for lag in range(0, max_lag + 1):
            # Shift `y` based on the current lag
            # if lag < 0:
            #     y_lagged = y[-lag:]  # Take the last part of y
            #     x_lagged = x[:len(y_lagged)]  # Adjust x to match y_lagged
            if lag > 0:
                y_lagged = y[:-lag]  # Take all but the last `lag` values
                x_lagged = x[lag:]  # Adjust x to match y_lagged
            else:
                y_lagged = y
                x_lagged = x
            
            # Check if lengths match after lagging
            if len(x_lagged) == len(y_lagged) and len(x_lagged) > 0:
                tau, p_value = kendalltau(x_lagged, y_lagged)
                # Update best lag if this one is better
                if best_tau is None or tau > best_tau:
                    best_tau = tau
                    best_p_value = p_value
                    best_lag = lag

        return best_tau, best_p_value, best_lag

    # # Calculate cross-correlation
    # max_corr, best_lag = cross_correlation(vect_1, vect_2)
    # metrics['max_corr'] = max_corr
    # metrics['best_lag'] = best_lag

    # # Calculate Phase Locking Value
    # plv = phase_locking_value(vect_1, vect_2)
    # #metrics['PLV'] = plv

    # # Calculate Time-lagged Mutual Information for the best lag
    # mi = time_lagged_mutual_information(vect_1, vect_2, lag=best_lag)
    # metrics['mutual_information'] = mi

    # # Calculate Instantaneous Phase Difference
    # phase_diff = instantaneous_phase_difference(vect_1, vect_2)
    # #metrics['instantaneous_phase_diff'] = phase_diff

    # Calculate Kendall's Tau
    best_tau, best_p_value, best_lag = kendall_tau_with_lags(vect_1, vect_2, apply_lag=apply_lag)
    #metrics['kendall_tau'] = {'tau': tau, 'p_value': p_value}
    metrics['kendall_tau'] = best_tau
    metrics['best_lag'] = best_lag

    return metrics


def last_decreasing_point(data, N=5):
    # Check if the array is large enough for the comparison
    if len(data) < N:
        return 0  # Not enough data points for N steps
    
    # Iterate backwards through the data
    for i in range(len(data) - N):
        # Check if the next N points are monotonically decreasing
        if all(data[j] > data[j+1] for j in range(i, i + N)):
            return i  # Return the last point where it's monotonically decreasing for N steps
    
    return 0  # If no such point is found

def local_max(time_series, patinent=10):
    n = len(time_series)
    max_value = torch.tensor(-float('inf'))
    counter = 0
    max_position = 0

    for i, t in enumerate(time_series):
        if t > max_value:
            max_value = t
            counter = 0
            max_position = i
        else:
            counter += 1
        
        # Break the loop if patience is exceeded
        if counter >= patinent:
            break

    return max_position

def local_min(time_series, patinent=10):
    n = len(time_series)
    min_value = torch.tensor(float('inf'))
    counter = 0
    min_position = 0

    for i, t in enumerate(time_series):
        if t < min_value:
            min_value = t
            counter = 0
            min_position = i
        else:
            counter += 1
        
        # Break the loop if patience is exceeded
        if counter >= patinent:
            break

    return min_position

def last_local_max(time_series):
    n = len(time_series)

    if n < 3:
        return n-1  # Not enough points to find a local minimum

    for i in range(n - 2, 0, -1):
        if time_series[i - 1] < time_series[i] > time_series[i + 1]:
            return i
    
    return n-1  #Return last point

def last_local_min(time_series):
    n = len(time_series)

    if n < 3:
        return n-1  # Not enough points to find a local minimum

    for i in range(n - 2, 0, -1):
        if time_series[i - 1] > time_series[i] < time_series[i + 1]:
            return i
    
    return n-1  # Return last point

class Performances:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.train_loss = []
        self.train_acc = []
        self.test_acc = []
        self.misslab_acc = []
        self.known_polluted_accuracy_loglik = []
        self.known_polluted_accuracy = []
        self.CSR = []
        self.PC = []
        self.LS = []
        self.CGE = []
        self.CND = []
        self.CND_noisy = []

        #self.PMFs_to_plot = []

        self.train_accuracy_mask = []
        self.test_accuracy_mask = []
        # Initialize storage for gradient alignment lists
        self.GA = []

    def update(self, train_loss, train_acc, test_acc, misslab_acc, known_polluted_accuracy_loglik, known_polluted_accuracy, performances_dict, args):

        def get_performance(performances_dict, attr):
            if attr in performances_dict:
                return performances_dict[attr]
            else:
                return None
            
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.test_acc.append(test_acc)
        self.misslab_acc.append(misslab_acc)
        self.known_polluted_accuracy_loglik.append(known_polluted_accuracy_loglik)
        self.known_polluted_accuracy.append(known_polluted_accuracy)

        self.CSR.append(get_performance(performances_dict, "CSR"))
        self.PC.append(get_performance(performances_dict, "PC"))
        self.LS.append(get_performance(performances_dict, "LS"))
        self.CGE.append(get_performance(performances_dict, "CGE"))

        CNDs = []
        CND_noisy = []

        for CND_type in args.CND_type:
            CNDs.append(get_performance(performances_dict, f"CND_{CND_type}"))
            CND_noisy.append(get_performance(performances_dict, f"CND_noisy_{CND_type}"))

        if "CND" in args.metrics:
            self.CND.append(torch.stack(CNDs))
        if "CND_noisy" in args.metrics:
            self.CND_noisy.append(torch.stack(CND_noisy))

        self.train_accuracy_mask.append(get_performance(performances_dict, "train_accuracy_mask"))
        self.test_accuracy_mask.append(get_performance(performances_dict, "test_accuracy_mask"))

        # Save GA_list per epoch, extend if list, else append None
        ga_values = performances_dict.get("GA_list", None)
        if ga_values is not None:
            self.GA.extend(ga_values)
        else:
            self.GA.append(None)


    def __len__(self):
        return len(self.train_acc)
    
    def torch_transformation(self):
        self.test_acc = torch.tensor(self.test_acc)

        if not is_all_none(self.CSR):
            self.CSR = torch.tensor(self.CSR)

        if not is_all_none(self.PC):
            self.PC = torch.tensor([x for x in self.PC if x is not None])

        if not is_all_none(self.CND):
            self.CND = torch.stack(self.CND)
            self.CND = self.CND.transpose(0, 1)


        if not is_all_none(self.CND_noisy):
            self.CND_noisy = torch.stack(self.CND_noisy)
            self.CND_noisy = self.CND_noisy.transpose(0, 1)


    def evaluate_memorization_metrics(self, args, window_size=1):
        best_epoch = torch.argmax(self.test_acc)

        # Initialize an empty list to collect each technique's performance
        performances = []

        # General performances
        gen_performance = {"Technique": "Test_acc", "best_epoch": best_epoch.item(), "Accuracy": self.test_acc[best_epoch].item()}
        performances.append(gen_performance)

        # Check if PC exists
        if "PC" in args.metrics:         #if not is_all_none(self.PC):
            PC_mav = self.moving_average(self.PC, window_size=1)
            PC_best_epoch = local_min(PC_mav)
            PC_performance = {"Technique": "PC", "best_epoch": PC_best_epoch, "Accuracy": self.test_acc[PC_best_epoch].item()}
            performances.append(PC_performance)


        # Check if CND exists
        if "CND" in args.metrics:    #if not is_all_none(self.CND):
            # for i in range(self.CND.shape[1]):
            #     CND_mav = self.moving_average(self.CND[:, i], window_size=window_size)
            #     CND_corr_metrics = last_decreasing_point(CND_mav)
            #     PC_performance = {"Technique": "PC", "best_epoch": PC_best_epoch, "Accuracy": self.test_acc[PC_best_epoch].item()}

            #     df_performances.append(CND_corr_metrics)

            CND_mean = torch.mean(self.CND, axis=1)
            CND_mean_mav = self.moving_average(CND_mean, window_size=window_size)
            CND_mean_best_epoch = local_max(CND_mean_mav)
            CND_mean_performance = {"Technique": f"CND_mean", "best_epoch": CND_mean_best_epoch, "Accuracy": self.test_acc[CND_mean_best_epoch].item()}
            performances.append(CND_mean_performance)


        # Check if PC exists
        if "CSR" in args.metrics:  #if not is_all_none(self.CSR):
            CSR_mav = self.moving_average(self.CSR, window_size=window_size)
            CSR_best_epoch = local_min(CSR_mav)
            CSR_performance = {"Technique": "CSR", "best_epoch": CSR_best_epoch, "Accuracy": self.test_acc[CSR_best_epoch].item()}
            performances.append(CSR_performance)

        # Convert the list of performances into a Pandas DataFrame
        df_performances = pd.DataFrame(performances)

        return df_performances



    def evaluate_correlation_metrics(self, args):

        def compute_corr_perf(performances_corr_list, vect_1, vect_2, name_vect_1, name_vect_2):

            performance_corr = compute_correlation_metrics(vect_1, vect_2)
            performance_corr['vect_1'] = name_vect_1 #f"CND_l{l_idx}"
            performance_corr['vect_2'] = name_vect_2 #f"test_acc"
            performances_corr_list.append(performance_corr)
            return performances_corr_list
        
        # Initialize an empty list to collect each technique's performance
        performances_corr_list = []

        performances_corr_list = compute_corr_perf(performances_corr_list, 
                                                   self.test_acc, self.misslab_acc, 
                                                   "test_acc", "misslab_acc")

        
        # Check if PC exists
        if "PC" in args.metrics:         #if not is_all_none(self.PC):

            performances_corr_list = compute_corr_perf(performances_corr_list, 
                                                    self.moving_average(self.PC, window_size=1), self.test_acc, 
                                                    "PC", "test_acc")
            
            performances_corr_list = compute_corr_perf(performances_corr_list, 
                                                    self.moving_average(self.PC, window_size=1), self.misslab_acc, 
                                                    "PC", "misslab_acc")
            
        # Check if CND exists
         
        start_idxes = [n_neurs_l(l_idx, args.neurs_x_hid_lyr) for l_idx in range(len(args.neurs_x_hid_lyr.items()))] 
        end_idxes = [n_neurs_l(l_idx+1, args.neurs_x_hid_lyr) for l_idx in range(len(args.neurs_x_hid_lyr.items()))] 
        
        if "CND" in args.metrics:    #if not is_all_none(self.CND):
            for l_idx in range(len(start_idxes)):

                start_idx = start_idxes[l_idx]
                end_idx = end_idxes[l_idx]

                CND_data = torch.mean(self.CND[:, start_idx:end_idx], axis=1)
    
                performances_corr_list = compute_corr_perf(performances_corr_list, 
                                                            CND_data, self.test_acc, 
                                                            f"CND_l{l_idx}", "test_acc")
            
                performances_corr_list = compute_corr_perf(performances_corr_list, 
                                                            CND_data, self.misslab_acc, 
                                                            f"CND_l{l_idx}", "misslab_acc")


        # Check if PC exists
        # if "CSR" in args.metrics:  #if not is_all_none(self.CSR):

        #     CSR_performance_corr = compute_correlation_metrics(self.moving_average(self.CSR, window_size=1), self.test_acc)
        #     CSR_performance_corr['metric'] = "CSR"
        #     performances_corr.append(CSR_performance_corr)

        #     CSR_mav = self.moving_average(self.CSR, window_size=1)
        #     CSR_best_epoch = local_min(CSR_mav)
        #     CSR_performance = {"Technique": "CSR", "best_epoch": CSR_best_epoch, "Accuracy": self.test_acc[CSR_best_epoch].item()}
        #     performances_corr.append(CSR_performance)

        # Convert the list of performances into a Pandas DataFrame
        self.df_performances_corr = pd.DataFrame(performances_corr_list)

        return None

    

    @staticmethod
    def moving_average(data, window_size=3):
        """Computes the moving average for a given list attribute."""
        if len(data) < window_size:
            return None  # Not enough data to compute moving average
        if window_size > 1:
            return torch.tensor([sum(data[i:i + window_size]) / window_size for i in range(len(data) - window_size + 1)])
        else:
            return torch.tensor(data)  # Ensure the return type is a tensor
    


    def plot_data_with_moving_averages(self, data_name, function_type, epochs, window_sizes, color, args):

        data = getattr(self, data_name)
        labels = [f'{data_name}_{i+1}' for i in range(len(window_sizes))]
        
        data_normalized = data / torch.max(data, axis=0)[0]

        plt.figure(figsize=(12, 8))

        cmap = plt.cm.get_cmap(color)
        colors = cmap(np.linspace(0.3, 0.9, len(window_sizes)))  # Skip the lightest greens

        for i, window_size in enumerate(window_sizes):
            if window_size == 1:
                plot_data = data_normalized
            else:
                plot_data = self.moving_average(data_normalized, window_size=window_size)
            
            plt.plot(epochs[i:], plot_data, label=labels[i], color=colors[i], marker='x', linestyle='--')
            local_extrema_idx = function_type(plot_data) + window_size
            plt.axvline(x=local_extrema_idx, color=colors[i], linestyle=':', linewidth=2)
            print(f"Window size {window_size}, Last local extrema at epoch {local_extrema_idx}")

        # Customize the plot
        plt.ylabel(f'{data_name}', fontsize=12)
        plt.tick_params(axis='y')#, labelcolor=color)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlabel('Epoch', fontsize=12)
        #plt.title('Average Data Over Epochs', fontsize=14)
        plt.legend(loc='best', fontsize=12)

        # Save the plot
        plt.tight_layout()
        plt.savefig(f"{args.results_dir}/{data_name}_{args.timestamp}.png")
        plt.close()


    def plot_performances(self, args):
        """
        Plot training performances including train/test accuracy, train loss, and other metrics.
        """

        def plot_accuracy_curves():
            """Plot train, test, corrupted accuracy, and expected mislabeled accuracy."""
            epochs = list(range(1, len(self.train_acc) + 1))
            expected_misslab_acc = [(1 - acc) / 9 for acc in self.test_acc]
            max_test_acc_index = torch.argmax(self.test_acc).item()

            plt.figure(figsize=(12, 8))
            ax1 = plt.gca()
            line1, = ax1.plot(epochs, self.train_acc, label='Train Accuracy', marker='o', linestyle='-')
            line2, = ax1.plot(epochs, self.test_acc, label='Test Accuracy', marker='o', linestyle='-')
            if any(self.misslab_acc):
                line3, = ax1.plot(epochs, self.misslab_acc, label='Corrupted Accuracy', marker='o', linestyle='-')
            line5 = ax1.axvline(x=epochs[max_test_acc_index], color='red', linestyle='--', label='Max Test Accuracy')
            if not all(x is None for x in self.known_polluted_accuracy):
                line4, = ax1.plot(epochs, self.known_polluted_accuracy, label='Polluted Accuracy Known', color='green', marker='o', linestyle='--')

            ax1.set_xlabel('Epoch', fontsize=16)
            ax1.set_ylabel('Accuracy', fontsize=16)
            ax1.grid(True, linestyle='--', alpha=0.6)
            ax1.tick_params(axis='y')

            ax2 = ax1.twinx()
            line6, = ax2.plot(epochs, self.train_loss, label='Train Loss', color='orange', marker='s', linestyle='-')
            ax2.set_ylabel('Train Loss', fontsize=16, color='purple')
            ax2.tick_params(axis='y', labelcolor='purple')

            lines = [line1, line2, line5, line6]
            labels = ['Train Accuracy', 'Test Accuracy', 'Max Test Accuracy', 'Train Loss']
            if any(self.misslab_acc):
                lines.insert(2, line3)
                labels.insert(2, 'Corrupted Accuracy')
            if not all(x is None for x in self.known_polluted_accuracy):
                lines.insert(3, line4)
                labels.insert(3, 'Polluted Accuracy Known')
            ax1.tick_params(axis='both', labelsize=14)
            ax2.tick_params(axis='both', labelsize=14)
            plt.legend(lines, labels, loc='best', fontsize=16)
            plt.xlim(epochs[0], epochs[-1])
            plt.tight_layout()
            plt.savefig(f"{args.results_dir}/training_curves_{args.timestamp}.png")
            plt.close()
            

        def plot_moving_average(data_name, extrema_function, color):
            """Plot data with moving averages and highlight extrema."""
            data = getattr(self, data_name)
            epochs = list(range(1, len(data) + 1))
            window_sizes = [1, 2, 3]
            cmap = plt.cm.get_cmap(color)
            colors = cmap(np.linspace(0.3, 0.9, len(window_sizes)))

            plt.figure(figsize=(12, 8))
            for i, window_size in enumerate(window_sizes):
                if window_size > 1:
                    plot_data = self.moving_average(data, window_size=window_size)
                else:
                    plot_data = data
                plt.plot(epochs[:len(plot_data)], plot_data, label=f'{data_name}_{window_size}', color=colors[i])
                extrema_index = extrema_function(plot_data)
                plt.axvline(x=extrema_index, color=colors[i], linestyle=':', linewidth=2)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel(data_name, fontsize=12)
            plt.legend(loc='best', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(f"{args.results_dir}/{data_name}_moving_average_{args.timestamp}.png")
            plt.close()

        def plot_CND(CND_data, data_label):
            """
            Plot CND metrics and their gradients for the given CND data.

            Args:
                CND_data (numpy.ndarray): The data to plot. Should have the same shape as self.CND or self.CND_noisy.
                data_label (str): A label to differentiate the dataset (e.g., 'CND' or 'CND_noisy').
            """
            epochs = list(range(1, len(self.train_acc) + 1))
            start_idxes = [n_neurs_l(l_idx, args.neurs_x_hid_lyr) for l_idx in range(len(args.neurs_x_hid_lyr.items()))]
            end_idxes = [n_neurs_l(l_idx + 1, args.neurs_x_hid_lyr) for l_idx in range(len(args.neurs_x_hid_lyr.items()))]

            for ii, CND_type in enumerate(args.CND_type):
                # Plot CND metrics
                plt.figure(figsize=(12, 8))
                for l_idx, (start_idx, end_idx) in enumerate(zip(start_idxes, end_idxes)):
                    layer_data = CND_data[ii, :, start_idx:end_idx]
                    mean_layer_data = layer_data.mean(axis=1)
                    std_layer_data = layer_data.std(axis=1)
 
                    plt.plot(epochs, mean_layer_data, label=f'Layer {l_idx}', linewidth=5)
                    num_steps = 50
                    for step in range(num_steps):
                        start_alpha = 0.05  # lower starting alpha
                        alpha = start_alpha * (1 - step / num_steps)
                        plt.fill_between(
                            epochs,
                            mean_layer_data - std_layer_data * step / num_steps,
                            mean_layer_data + std_layer_data * step / num_steps,
                            color=plt.gca().lines[-1].get_color(),
                            alpha=alpha
                        )
                plt.xlim(epochs[0], epochs[-1])
                plt.xlabel('Epoch', fontsize=16)
                plt.ylabel('CND', fontsize=16)
                plt.legend(loc='upper right', fontsize=16)
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                plt.tick_params(labelsize=14)
                plt.savefig(f"{args.results_dir}/{data_label}_{CND_type}_layer_{l_idx}_CND_{args.timestamp}.png")
                plt.close()

                # Plot gradients of CND metrics
                plt.figure(figsize=(12, 8))
                for l_idx, (start_idx, end_idx) in enumerate(zip(start_idxes, end_idxes)):
                    layer_data = CND_data[ii, :, start_idx:end_idx]
                    gradient = np.gradient(layer_data, axis=0)
                    mean_layer_data = gradient.mean(axis=1)
                    std_layer_data = gradient.std(axis=1)
 
                    plt.plot(epochs, mean_layer_data, label=f'Layer {l_idx} Gradient {CND_type}', linewidth=5)
                    num_steps = 50
                    for step in range(num_steps):
                        start_alpha = 0.05  # lower starting alpha
                        alpha = start_alpha * (1 - step / num_steps)
                        plt.fill_between(
                            epochs,
                            mean_layer_data - std_layer_data * step / num_steps,
                            mean_layer_data + std_layer_data * step / num_steps,
                            color=plt.gca().lines[-1].get_color(),
                            alpha=alpha
                        )
                plt.xlim(epochs[0], epochs[-1])
                plt.xlabel('Epoch', fontsize=16)
                plt.ylabel('Gradient of CND', fontsize=16)
                plt.legend(loc='upper right', fontsize=16)
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                plt.tick_params(labelsize=14)
                plt.savefig(f"{args.results_dir}/{data_label}_{CND_type}_layer_{l_idx}_Gradient_{args.timestamp}.png")
                plt.close()

        def plot_cnd_quartiles(CND_data, data_label):
            """
            Plots the top and bottom quartiles of CND for each layer, with distinct colors for each layer
            and gradient variations for quartile curves.

            Args:
                CND_data (torch.Tensor): The data to plot. Should be either self.CND or self.CND_noisy.
                data_label (str): A label to differentiate the dataset (e.g., 'CND' or 'CND_noisy').
            """
            results_dir = args.results_dir
            epochs = list(range(CND_data.shape[1]))  # Assuming epochs are along the first dimension
            start_idxes = [n_neurs_l(l_idx, args.neurs_x_hid_lyr) for l_idx in range(len(args.neurs_x_hid_lyr.items()))]
            end_idxes = [n_neurs_l(l_idx + 1, args.neurs_x_hid_lyr) for l_idx in range(len(args.neurs_x_hid_lyr.items()))]

            def plot_quartiles(quartiles_func, title, filename_suffix):
                """
                Plots the quartiles of CND for each layer with a single legend entry for each layer,
                using distinct colors for each layer and gradient variations for quartile curves.

                Args:
                    quartiles_func (callable): A function to compute quartiles.
                    title (str): The title of the plot.
                    filename_suffix (str): A suffix for the output filename.
                """
                for ii, CND_type in enumerate(args.CND_type):
                    plt.figure(figsize=(12, 8))

                    cmap = plt.cm.get_cmap("tab10")  # Use a colormap for layer colors
                    layer_colors = cmap(np.linspace(0, 1, len(start_idxes)))  # Generate distinct colors for each layer

                    for l_idx, (start_idx, end_idx) in enumerate(zip(start_idxes, end_idxes)):
                        layer_data = CND_data[ii, :, start_idx:end_idx].numpy()  # Extract CND values for the layer
                        layer_quartiles = quartiles_func(layer_data)  # Compute quartiles for the layer
                        
                        # Plot all quartiles for the layer with the same color but varying transparency
                        for q_idx, quartile_data in enumerate(layer_quartiles):
                            alpha = 0.3 + 0.7 * (q_idx / (len(layer_quartiles) - 1))  # Gradual transparency
                            plt.plot(epochs, quartile_data, color=layer_colors[l_idx], alpha=alpha, linestyle='-')

                        # Add a single legend entry for the layer with its color
                        plt.plot([], [], color=layer_colors[l_idx], label=f'Layer {l_idx}')

                    plt.title(title, fontsize=14)
                    plt.xlabel('Epoch', fontsize=12)
                    plt.ylabel('CND', fontsize=12)
                    plt.grid(True, linestyle='--', alpha=0.6)
                    plt.legend(loc='best', fontsize=10)
                    plt.tight_layout()
                    plt.savefig(f"{results_dir}/{data_label}_{CND_type}_{filename_suffix}_{args.timestamp}.png")
                    plt.close()

            def compute_top_quartiles(layer_data):
                """
                Compute the top quartiles for each epoch.
                """
                return np.percentile(layer_data, [50, 75, 90, 95, 99, 99.5, 100], axis=1)

            def compute_bottom_quartiles(layer_data):
                """
                Compute the bottom quartiles for each epoch.
                """
                return np.percentile(layer_data, [0, 1, 2, 5, 10, 25, 40], axis=1)

            # Plot Top Quartiles
            plot_quartiles(compute_top_quartiles, "Top Quartiles", "top_quartiles")

            # Plot Bottom Quartiles
            plot_quartiles(compute_bottom_quartiles, "Bottom Quartiles", "bottom_quartiles")


        def plot_class_graph_entropy():
            """Plot Class Graph Entropy metrics."""
            CGE = torch.tensor(self.CGE)
            epochs = list(range(1, CGE.shape[0] + 1))

            plt.figure(figsize=(12, 8))
            for i in range(CGE.shape[1]):
                plt.plot(epochs, CGE[:, i], label=f'Class {i}')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Class Graph Entropy', fontsize=12)
            plt.legend(loc='best', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(f"{args.results_dir}/CGE_{args.timestamp}.png")
            plt.close()

        # Call subfunctions
        plot_accuracy_curves()

        if "CSR" in args.metrics:
            plot_moving_average("CSR", local_max, 'Blues')


        if "PC" in args.metrics:
            plot_moving_average("PC", local_min, 'Purples')

        if "CND" in args.metrics:
            plot_CND(self.CND, "CND")
            plot_cnd_quartiles(self.CND, "CND")

        if "CND_noisy" in args.metrics:
            plot_CND(self.CND_noisy, "CND_noisy")
            plot_cnd_quartiles(self.CND_noisy, "CND_noisy")

        if "CGE" in args.metrics:
            plot_class_graph_entropy()


    def plot_PMFs(self, args):

        # Create a directory for saving plots if it doesn't exist
        results_dir = args.results_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        if not os.path.exists(f"{results_dir}/PMFs_images"):
            os.makedirs(f"{results_dir}/PMFs_images")

        for PMF in self.PMFs_to_plot:

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
    def get_flattened_GA(self, key=0):
        """
        Extracts and flattens the GA values from the list of dictionaries stored in self.GA.

        Parameters:
        - key (int): The key to extract values from each dictionary (default is 0).

        Returns:
        - list of float: Flattened list of GA values.
        """
        return [d[key] for d in self.GA if d is not None and key in d]