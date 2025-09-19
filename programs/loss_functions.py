import torch
import torch.nn as nn

def one_hot_encode(targets, num_classes):
    """
    Converts integer class labels to one-hot encoded vectors.
    """
    # Ensure targets are int64 and 1D
    if targets.ndim > 1:
        targets = targets.view(-1)  # Flatten if needed
    targets = targets.long()  # Convert to int64 for scatter

    # Create the one-hot tensor
    one_hot = torch.zeros(targets.size(0), num_classes, device=targets.device)
    one_hot.scatter_(1, targets.unsqueeze(1), 1)
    return one_hot

def define_loss_function(args):
    """
    Define the loss function based on the argument `args.loss_function`.
    """
    if args.loss_function == "cross_entropy_loss":
        criterion = nn.CrossEntropyLoss()

    elif args.loss_function == "MAE":
        def mae_loss_with_logits(outputs, targets, num_classes):
            """
            Custom MAE loss function that expects logits as outputs and integer labels as targets.
            The labels are one-hot encoded before computing the loss.
            """
            # One-hot encode the targets
            one_hot_targets = one_hot_encode(targets, num_classes)
            
            # Convert logits to probabilities
            probs = torch.softmax(outputs, dim=1)
            
            # Compute the Mean Absolute Error
            return nn.L1Loss()(probs, one_hot_targets)

        # Define the criterion as a lambda for easier usage
        criterion = lambda outputs, targets: mae_loss_with_logits(outputs, targets, args.num_classes)

    else:
        raise ValueError(f"Unsupported loss function: {args.loss_function}")

    return criterion