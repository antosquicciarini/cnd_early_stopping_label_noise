import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt


class DatasetDefinition(Dataset):
    """
    A dataset wrapper that allows label perturbation, tracks indices,
    provides added functionality, and can inject random noise into the dataset.
    Compatible with both preloaded (in-memory) datasets (which have a 'data'
    attribute) and lazy-loaded datasets (e.g., torchvision.datasets.ImageFolder).
    """
    def __init__(self, dataset, num_classes, perturb_ratio=0.0, transform=None):
        """
        Initializes the dataset definition.

        Args:
            dataset (torch.utils.data.Dataset): The base dataset.
            num_classes (int): The number of classes in the dataset.
            perturb_ratio (float): The percentage of labels to perturb on each call.
            transform (callable, optional): A function/transform to apply to the data.
        """
        self.num_classes = num_classes
        self.perturb_ratio = perturb_ratio

        # Check if the dataset is preloaded (has attribute 'data') or lazy-loading
        if hasattr(dataset, 'data'):
            self.lazy = False

            if dataset.data.dtype != torch.int32: #No text data
                # If data is a NumPy array, convert it to a tensor.
                if isinstance(dataset.data, np.ndarray):
                    self.data = torch.tensor(dataset.data, dtype=torch.float32)
                else:
                    self.data = dataset.data
                # If images are in [N, H, W, C] format, permute to [N, C, H, W]
                if self.data.ndim == 4 and self.data.shape[-1] == 3:
                    self.data = self.data.permute(0, 3, 1, 2)
                # For grayscale images [N, H, W] add a channel dimension.
                elif self.data.ndim == 3:
                    self.data = self.data.unsqueeze(1)

                # Normalize if pixel values exceed 1.
                if self.data.max() > 1:
                    self.data = self.data / 255.0
            else: 
                self.data = dataset.data

            self.targets = torch.tensor(dataset.targets)
            # To track new samples if expanding the dataset.
            self.new_indexes = []

            
        else:
            self.lazy = True
            self.dataset = dataset  # e.g. an ImageFolder instance
            self.original_length = len(dataset)
            self.targets = torch.tensor(dataset.targets)
            # For lazy datasets, we store in-place modifications in a dictionary.
            # Key: index, Value: (modified_image, modified_label). If modified_image is None,
            # then the image remains as originally loaded.
            self.noisy_modifications = {}
            # For new (expanded) samples, we cache them as (image, label) tuples.
            self.new_data = []
            self.new_indexes = []

        # If the dataset is preloaded, you may not want to apply ToTensor() (since data is already a tensor).
        # In lazy mode, you likely need it to convert PIL images.
        if not self.lazy and transform and hasattr(transform, 'transforms'):
            self.transform = transforms.Compose(
                [t for t in transform.transforms if not isinstance(t, transforms.ToTensor)]
            )
        else:
            self.transform = transform

    def __getitem__(self, index):
        """
        Returns the data, label, and index.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: (data, label, index)
        """
        if not self.lazy:
            data, label = self.data[index], self.targets[index]
            # Apply random label perturbation
            if random.random() < self.perturb_ratio:
                label = random.choice([cls for cls in range(self.num_classes) if cls != label])
            if self.transform:
                data = self.transform(data)
            return data, label, index
        else:
            # For lazy datasets, first determine whether the requested index
            # is from the original dataset or from newly added (expanded) samples.
            if index < self.original_length:
                if index in self.noisy_modifications:
                    mod_data, mod_label = self.noisy_modifications[index]
                    # If no image modification was stored, load the original image.
                    if mod_data is None:
                        data, label = self.dataset[index]
                    else:
                        data, label = mod_data, mod_label
                else:
                    data, label = self.dataset[index]
            else:
                # For expanded samples, the index is shifted.
                new_index = index - self.original_length
                data, label = self.new_data[new_index]

            # Apply random label perturbation
            if random.random() < self.perturb_ratio:
                label = random.choice([cls for cls in range(self.num_classes) if cls != label])
            # if self.transform:
            #     data = self.transform(data)
            return data, label, index

    def __len__(self):
        if not self.lazy:
            return len(self.targets)
        else:
            return self.original_length + len(self.new_data)

    def add_polluted_instances_to_images(
        self,
        noise_ratio=0.1,
        expand=False,
        symmetrical_label_noise=False,
        subset_indices=None
    ):
        """
        Injects random Gaussian-like noise OR applies symmetrical label noise in-place
        to this dataset. Can optionally draw from a subset of indices only.

        Args:
            noise_ratio (float):
                Proportion of samples to replace or add as noise/label corruption.
            expand (bool):
                - If False, corrupts (overwrites) samples in-place.
                - If True, appends noisy samples (thus expanding the dataset).
            symmetrical_label_noise (bool):
                - If False, applies random noise to the images.
                - If True, applies label corruption (images remain unchanged).
            subset_indices (list or torch.Tensor, optional):
                If provided, only these indices are considered for corruption.
        
        Returns:
            torch.Tensor:
                A 1D tensor of indices that were corrupted.
                For expand=False, these refer to the samples that were replaced;
                for expand=True, these refer to the newly added samples.
        """
        if not self.lazy:
            # ----- Preloaded mode -----
            num_total_samples = len(self.data)
            if subset_indices is not None:
                subset_indices = list(subset_indices) if not isinstance(subset_indices, list) else subset_indices
                num_available = len(subset_indices)
            else:
                subset_indices = list(range(num_total_samples))
                num_available = num_total_samples

            num_noise_samples = int(noise_ratio * num_available)
            if num_noise_samples == 0:
                return torch.tensor([])

            if not expand:
                chosen_indices = random.sample(subset_indices, num_noise_samples)
                noise_indices = torch.tensor(chosen_indices)
            else:
                noise_indices = torch.arange(num_total_samples, num_total_samples + num_noise_samples)
                chosen_indices = [random.choice(subset_indices) for _ in range(num_noise_samples)]

            if symmetrical_label_noise:
                if not expand:
                    for idx in noise_indices:
                        original_label = self.targets[idx].item() if hasattr(self.targets[idx], "item") else self.targets[idx]
                        noisy_label = random.choice([cls for cls in range(self.num_classes) if cls != original_label])
                        self.targets[idx] = noisy_label
                else:
                    new_data = []
                    new_labels = []
                    for source_idx in chosen_indices:
                        original_label = self.targets[source_idx].item() if hasattr(self.targets[source_idx], "item") else self.targets[source_idx]
                        noisy_label = random.choice([cls for cls in range(self.num_classes) if cls != original_label])
                        sample_data = self.data[source_idx].clone() if hasattr(self.data[source_idx], "clone") else self.data[source_idx]
                        new_data.append(sample_data)
                        new_labels.append(noisy_label)
                    new_data = torch.stack(new_data)
                    new_labels = torch.tensor(new_labels, dtype=self.targets.dtype)
                    self.data = torch.cat((self.data, new_data), dim=0)
                    self.targets = torch.cat((self.targets, new_labels), dim=0)
                    self.new_indexes.extend(noise_indices.tolist())
                return noise_indices
            else:
                # ----- Random Gaussian-like noise for preloaded data -----
                if not expand:
                    for idx in noise_indices:
                        random_noise_image = torch.randint_like(
                            self.data[idx],
                            low=0,
                            high=256,
                            dtype=self.data[idx].dtype
                        )
                        if self.data[idx].dtype != torch.uint8:
                            random_noise_image = random_noise_image.float() / 255.0
                        self.data[idx] = random_noise_image
                else:
                    new_data = []
                    new_labels = []
                    for source_idx in chosen_indices:
                        sample_shape = self.data[source_idx]
                        random_noise_image = torch.randint_like(
                            sample_shape,
                            low=0,
                            high=256,
                            dtype=sample_shape.dtype
                        )
                        if sample_shape.dtype != torch.uint8:
                            random_noise_image = random_noise_image.float() / 255.0
                        random_label = random.randint(0, self.num_classes - 1)
                        new_data.append(random_noise_image)
                        new_labels.append(random_label)
                    new_data = torch.stack(new_data)
                    new_labels = torch.tensor(new_labels, dtype=self.targets.dtype)
                    self.data = torch.cat((self.data, new_data), dim=0)
                    self.targets = torch.cat((self.targets, new_labels), dim=0)
                    self.new_indexes.extend(noise_indices.tolist())
                return noise_indices
        else:
            # ----- Lazy mode (e.g. ImageFolder) -----
            num_total_samples = self.original_length
            if subset_indices is not None:
                subset_indices = list(subset_indices) if not isinstance(subset_indices, list) else subset_indices
                num_available = len(subset_indices)
            else:
                subset_indices = list(range(num_total_samples))
                num_available = num_total_samples

            num_noise_samples = int(noise_ratio * num_available)
            if num_noise_samples == 0:
                return torch.tensor([])

            if not expand:
                chosen_indices = random.sample(subset_indices, num_noise_samples)
                noise_indices = torch.tensor(chosen_indices)
            else:
                noise_indices = torch.arange(num_total_samples, num_total_samples + num_noise_samples)
                chosen_indices = [random.choice(subset_indices) for _ in range(num_noise_samples)]

            if symmetrical_label_noise:
                if not expand:
                    for idx in chosen_indices:
                        # Load the original sample
                        _, original_label = self.dataset[idx]
                        noisy_label = random.choice([cls for cls in range(self.num_classes) if cls != original_label])
                        # Store modification with None for image (i.e. image remains unchanged)
                        self.noisy_modifications[idx] = (None, noisy_label)
                else:
                    for source_idx in chosen_indices:
                        img, original_label = self.dataset[source_idx]
                        noisy_label = random.choice([cls for cls in range(self.num_classes) if cls != original_label])
                        self.new_data.append((img, noisy_label))
                return noise_indices
            else:
                # ----- Random Gaussian-like noise for lazy datasets -----
                if not expand:
                    for idx in chosen_indices:
                        img, label = self.dataset[idx]
                        # Convert the PIL image to a NumPy array to determine its shape
                        arr = np.array(img)
                        # Generate random noise with the same shape and type
                        noise_arr = np.random.randint(0, 256, size=arr.shape, dtype=np.uint8)
                        # Convert the noise array back to a PIL image
                        noisy_img = Image.fromarray(noise_arr)
                        self.noisy_modifications[idx] = (noisy_img, label)
                else:
                    for source_idx in chosen_indices:
                        img, label = self.dataset[source_idx]
                        arr = np.array(img)
                        noise_arr = np.random.randint(0, 256, size=arr.shape, dtype=np.uint8)
                        noisy_img = Image.fromarray(noise_arr)
                        random_label = random.randint(0, self.num_classes - 1)
                        self.new_data.append((noisy_img, random_label))
                return noise_indices