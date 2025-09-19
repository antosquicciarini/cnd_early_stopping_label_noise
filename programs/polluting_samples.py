import torch
import random
from torch.utils.data import DataLoader, Subset
import torch
import random


# Function to generate random noise images
def add_random_gaussian_noise_to_images(dataset, noise_ratio=0.1):
    # Convert the dataset images to float for consistent processing
    #dataset.data = dataset.data.float()

    # Calculate the number of samples to replace with random noise
    num_samples = len(dataset)
    num_noise_samples = int(noise_ratio * num_samples)

    # Get random indices to replace
    noise_indices = torch.tensor(random.sample(range(num_samples), num_noise_samples))
    
    # Generate random noise images and replace
    for idx in noise_indices:
        random_noise_image = torch.randint_like(dataset.data[idx], low=0, high=256) #torch.rand_like(dataset.data[idx]) * 255 # Create random noise
        dataset.data[idx] = random_noise_image  # Replace image with noise
        
    return dataset, noise_indices


# Function to add random noise samples with random labels to the dataset
def add_random_gaussian_noise_to_images_expand_dataset(dataset, noise_ratio=0.1):
    # Calculate the number of samples to add
    num_samples = len(dataset)
    num_noise_samples = int(noise_ratio * num_samples)
    noise_indices = torch.arange(len(dataset), len(dataset)+num_noise_samples)

    # Create lists to store new data and labels
    new_data = []
    new_labels = []

    # Generate random noise images and random labels
    for _ in range(num_noise_samples):
        # Create a random noise image with the same shape as the dataset images
        random_noise_image = torch.randint(0, 256, dataset.data[0].shape, dtype=torch.uint8)
        # Generate a random label
        random_label = torch.randint(0, 10, (1,)).item()
        new_data.append(random_noise_image)
        new_labels.append(random_label)

    # Convert lists to tensors
    new_data = torch.stack(new_data)
    new_labels = torch.tensor(new_labels)
    
    # Combine with the original dataset
    dataset.data = torch.cat((dataset.data, new_data), dim=0)
    dataset.targets = torch.cat((dataset.targets, new_labels), dim=0)

    
    return dataset, noise_indices


def shuffle_labels(dataset, args, shuffle_ratio=0.3):
    """
    Shuffle the labels of a given percentage of the dataset. If args.LN_reg is True,
    extend the dataset instead of overwriting the labels, allowing the model to see
    both correct and noisy labels.

    Args:
        dataset (torchvision.datasets): The dataset whose labels will be shuffled.
        args: Arguments object containing LN_reg and other parameters.
        shuffle_ratio (float): The proportion of labels to shuffle (e.g., 0.3 for 30%).

    Returns:
        dataset: The dataset with shuffled labels (or extended if LN_reg is True).
    """
    num_samples = len(dataset)
    num_to_shuffle = int(num_samples * shuffle_ratio)

    if num_to_shuffle > num_samples:
        shuffled_indices = random.choices(range(num_samples), k=num_to_shuffle)
    else:
        shuffled_indices = random.sample(range(num_samples), num_to_shuffle) 

    if getattr(args, 'LN_reg', False):
        # Extend dataset with noisy labels
        new_data = []
        new_labels = []
        
        for idx in shuffled_indices:
            # Copy the data point
            noisy_sample = dataset.data[idx]
            # Generate a noisy label
            noisy_label = random.choices(
                [i for i in range(args.num_classes) if i != dataset.targets[idx]], k=1)[0]
            new_data.append(noisy_sample)
            new_labels.append(noisy_label)
        
        # Convert to tensors
        new_data = torch.stack(new_data)
        new_labels = torch.tensor(new_labels)
        
        # Extend the dataset
        dataset.data = torch.cat((dataset.data, new_data), dim=0)
        dataset.targets = torch.cat((dataset.targets, new_labels), dim=0)
        corrupted_samples = torch.arange(len(dataset.data) - len(new_data), len(dataset.data))
        not_corrupted_samples = torch.arange(len(dataset.data) - len(new_data))
        
        return dataset, corrupted_samples, not_corrupted_samples
    else:
        # Overwrite labels with noise
        for idx in shuffled_indices:
            dataset.targets[idx] = random.choices(
                [i for i in range(args.num_classes) if i != dataset.targets[idx]], k=1)[0]
            
        
        corrupted_samples = torch.tensor(shuffled_indices)
        mask = torch.tensor([i in shuffled_indices for i in range(num_samples)])
        not_corrupted_samples = torch.arange(num_samples)[~mask]
    
        return dataset, corrupted_samples, not_corrupted_samples


def hard_sample_memorization(dataset, args):
    shuffled_indices = []
    for excluded_class in args.excluded_classes:
        shuffled_indices.append(torch.where(torch.tensor(dataset.targets)==excluded_class)[0])
    shuffled_indices = torch.cat(shuffled_indices)
    for i, idx in enumerate(shuffled_indices):
        dataset.targets[idx]  = random.choices(
            [i for i in range(0, args.num_classes) if i not in args.excluded_classes], k=1)[0]
    args.num_classes = args.num_classes-len(args.excluded_classes)
    return dataset, shuffled_indices, args