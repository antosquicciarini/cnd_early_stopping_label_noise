import os
from torch.utils.data import Dataset
from PIL import Image
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle

# Torchvision related imports
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from tinyimgenet import load_tiny_imagenet

# Custom dataset and utility imports
from dataset_classes import DatasetDefinition
from dataset_classes_webvision import WebVisionDataset
from polluting_samples import (
    shuffle_labels,
    hard_sample_memorization,
    add_random_gaussian_noise_to_images,
    add_random_gaussian_noise_to_images_expand_dataset,
)
from news import load_glove_embeddings, regroup_dataset, NewsDataset


def load_dataset(args):
    # Fix the random seed for reproducibility
    if not hasattr(args, "seed"):
        args.seed = 18
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    corrupted_samples = []
    embedding_weights = None

    # =======================
    # MNIST and FashionMNIST
    # =======================
    if 'MNIST' in args.dataset:
        if args.dataset == 'MNIST':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # MNIST normalization
            ])
            transform_val = transform
            train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_val)
            args.input_channels = 1
            args.input_size = 28
            args.num_classes = 10
        elif args.dataset == 'FashionMNIST':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            transform_val = transform
            train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
            test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_val)
            args.input_channels = 1
            args.input_size = 28
            args.num_classes = 10
        corrupted_samples = torch.tensor([])
        not_corrupted_samples = torch.arange(len(train_dataset))

    # ===========
    # CIFAR-10
    # ===========
    elif args.dataset == 'CIFAR10':
        if getattr(args, 'DA', False):
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
        args.input_channels = 3
        args.input_size = 32
        args.num_classes = 10

        # Handle CIFAR-10N noisy label variants
        if args.noise_type in (
            "worse_label", "random_label1", "random_label2", "random_label3", "aggre_label", "clean_label"
        ):
            # See: https://github.com/UCSC-REAL/cifar-10-100n
            noise_file = torch.load('./data/cifarN_labels/CIFAR-10_human.pt', weights_only=False)
            clean_label = noise_file['clean_label']
            if args.noise_type == "worse_label":
                noisy_labels = noise_file['worse_label']
            elif args.noise_type == "random_label1":
                noisy_labels = noise_file['random_label1']
            elif args.noise_type == "random_label2":
                noisy_labels = noise_file['random_label2']
            elif args.noise_type == "random_label3":
                noisy_labels = noise_file['random_label3']
            elif args.noise_type == "aggre_label":
                noisy_labels = noise_file['aggre_label']
            elif args.noise_type == "clean_label":
                noisy_labels = noise_file['clean_label']
            noise_percentage = (1 - (torch.sum(torch.tensor(clean_label) == torch.tensor(noisy_labels)) / len(clean_label))) * 100
            print(f"CIFAR 10N - {args.noise_type} Selected")
            print(f"Label noise percentage: {noise_percentage:.3g}%")
            if not torch.equal(torch.tensor(noise_file['clean_label']), torch.tensor(train_dataset.targets)):
                raise ValueError("Labels uploaded are not the correct ones")
            train_dataset.targets = noisy_labels
            corrupted_samples = torch.arange(len(clean_label))[clean_label != noisy_labels]
            not_corrupted_samples = torch.arange(len(clean_label))[clean_label == noisy_labels]
        else:
            corrupted_samples = torch.tensor([])
            not_corrupted_samples = torch.arange(len(train_dataset))

    # ===========
    # CIFAR-100
    # ===========
    elif args.dataset == 'CIFAR100':
        if hasattr(args, 'DA') and args.DA:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)
        args.input_channels = 3
        args.input_size = 32
        args.num_classes = 100

        if args.noise_type in ("worse_label", "clean_label"):
            noise_file = torch.load('./data/cifarN_labels/CIFAR-100_human.pt')
            clean_label = noise_file['clean_label']
            if args.noise_type == "worse_label":
                noisy_labels = noise_file['noisy_label']
            elif args.noise_type == "clean_label":
                noisy_labels = noise_file['clean_label']
            noise_percentage = (1 - (torch.sum(torch.tensor(clean_label) == torch.tensor(noisy_labels)) / len(clean_label))) * 100
            print(f"CIFAR 100N - {args.noise_type} Selected")
            print(f"Label noise percentage: {noise_percentage:.3g}%")
            if not torch.equal(torch.tensor(noise_file['clean_label']), torch.tensor(train_dataset.targets)):
                raise ValueError("Labels uploaded are not the correct ones")
            train_dataset.targets = noisy_labels
            corrupted_samples = torch.arange(len(clean_label))[clean_label != noisy_labels]
            not_corrupted_samples = torch.arange(len(clean_label))[clean_label == noisy_labels]

    # ==================
    # Tiny ImageNet-200
    # ==================
    elif args.dataset == 'TinyImagenet':
        train_dataset, test_dataset, transform, transform_val = load_tiny_imagenet(args)
        args.input_channels = 3
        args.input_size = 64
        args.num_classes = 200

    # =====
    # SVHN
    # =====
    elif args.dataset == 'SVHN':
        if getattr(args, 'DA', False):
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            ])
        train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
        transform_val = transform
        train_dataset.targets = train_dataset.labels
        test_dataset.targets = test_dataset.labels
        args.input_channels = 3
        args.input_size = 32
        args.num_classes = 10

    # =====
    # NEWS
    # =====
    elif args.dataset == 'NEWS':
        # glove_path = "data/glove.6B/glove.6B.300d.txt" #we use the GloVe 300 version as Yu 19 and Yuan 24
        # vocab, embedding_weights = load_glove_embeddings(glove_path)

        embedding_weights, data, labels=pickle.load(open("data/20news-bydate/news.pkl", "rb"), encoding='iso-8859-1')
        labels = regroup_dataset(labels)

        length=labels.shape[0]

        training_data = torch.from_numpy(data[:int(length*0.70)])
        training_labels = torch.from_numpy(labels[:int(length*0.70)])

        test_data = torch.from_numpy(data[int(length*0.70):])
        test_labels = torch.from_numpy(labels[int(length*0.70):])
        

        # embedding_weights: [vocabulary_size, emb_dim]
        # No transforms for text datasets
        transform = None
        transform_val = None
        train_dataset = NewsDataset("train", training_data, training_labels) #NewsDataset("data/20news-bydate/20news-bydate-train", vocab)
        test_dataset = NewsDataset("test", test_data, test_labels) #NewsDataset("data/20news-bydate/20news-bydate-test", vocab)
        args.input_channels = 1
        args.input_size = args.max_length
        args.num_classes = 7
        #args.embedding_layer = torch.nn.Embedding.from_pretrained(weights, freeze=False)

    # ===============
    # mini-WebVision
    # ===============
    elif args.dataset == 'mini_webvision':
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if hasattr(args, 'DA') and args.DA:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transform_val
        train_dataset = WebVisionDataset(
            "./data/webvision/info/train_filelist_google.txt",
            "data/webvision/images",
            transform=transform
        )
        test_dataset = WebVisionDataset(
            "./data/webvision/info/val_filelist.txt",
            "data/webvision/images/val_images_256",
            transform=transform_val
        )
        args.input_channels = 3
        args.input_size = 224
        args.num_classes = 50

    # ==============================
    # Apply label noise if requested
    # ==============================
    if args.noise_type in ("label_noise", "hard_noise", "gaussian_noise"):
        if args.noise_type == "label_noise":
            train_dataset, corrupted_samples, not_corrupted_samples = shuffle_labels(
                train_dataset, args, shuffle_ratio=args.symmetric_label_noise_ratio
            )
        elif args.noise_type == "hard_noise":
            train_dataset, corrupted_samples, args = hard_sample_memorization(train_dataset, args)
        elif args.noise_type == "gaussian_noise":
            train_dataset, corrupted_samples = add_random_gaussian_noise_to_images(train_dataset, noise_ratio=0.3)

    # ===========================
    # Wrap datasets for training
    # ===========================
    train_dataset = DatasetDefinition(
        train_dataset,
        num_classes=args.num_classes,
        perturb_ratio=getattr(args, "perturb_ratio", 0.0),
        transform=transform
    )
    test_dataset = DatasetDefinition(
        test_dataset,
        num_classes=args.num_classes,
        transform=transform_val
    )

    # Optionally expand dataset with additional noise (CGA-known or expand_dataset options)
    if getattr(args, "CGA_known", False):
        _ = train_dataset.add_polluted_instances_to_images(
            noise_ratio=0.05,  # similar size as batch_size
            expand=True,
            symmetrical_label_noise=True,
            subset_indices=not_corrupted_samples
        )
        corrupted_samples += train_dataset.new_indexes
    if getattr(args, "expand_dataset", False):
        _ = train_dataset.add_polluted_instances_to_images(
            noise_ratio=args.expand_dataset_noise_ratio,
            expand=True,
            symmetrical_label_noise=args.expand_dataset_symmetric_ln,
            subset_indices=not_corrupted_samples
        )

    # Set iterations per epoch and sample indices for tracking
    args.iterations_per_epoch = (len(train_dataset) + args.batch_size - 1) // args.batch_size
    args.corrupted_samples = corrupted_samples.tolist()
    args.not_corrupted_samples = not_corrupted_samples.tolist()

    return train_dataset, test_dataset, embedding_weights, args