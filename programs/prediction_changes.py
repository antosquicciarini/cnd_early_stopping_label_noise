import numpy as np

def compute_changed_predictions(prev_predictions, current_predictions):
    """
    Compute the number of samples that changed their predictions between two epochs.
    
    Args:
        prev_predictions (torch.Tensor): Predictions from the previous epoch.
        current_predictions (torch.Tensor): Predictions from the current epoch.
        
    Returns:
        int: Number of samples with changed predictions.
    """
    # Convert tensors to numpy arrays for easier comparison
    prev_predictions = prev_predictions.cpu().numpy()
    current_predictions = current_predictions.cpu().numpy()
    
    # Compute the number of changed predictions
    changed = np.sum(prev_predictions != current_predictions)
    
    return changed
