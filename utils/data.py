from typing import Union, Tuple
import pickle
import os

import torch as T
import torch
import torch.nn as nn
from einops import rearrange, repeat
import numpy as np
from torchvision import datasets
from torch.utils.data import Dataset
from pathlib import Path
import torch.nn.functional as F




def get_dataset_presets():

    dataset_presets = {
            'cifar10_2cls': {
                'input_dim': 3*32*32,
                'output_dim': 1
            },
            'cifar10': {
                'input_dim': 3*32*32,
                'output_dim': 10
            },
            'cifar10_ez': {
                'input_dim': 3*32*32,
                'output_dim': 10
            },
            'svhn': {
                'input_dim': 3*32*32,
                'output_dim': 10
            },
            'fmnist': {
                'input_dim': 1*28*28,
                'output_dim': 10
            },
            'imagenet32': {
                'input_dim': 3*32*32,
                'output_dim': 1000
            },
            'shakespeare': {
                'input_dim': 128,
                'output_dim': 65,
            },
            'sst2': {
                'input_dim': 64,
                'output_dim': 1,
            },

        }

    return dataset_presets



def prepare_cifar10_2cls(dataset_folder: Path, num_data: int, classes: list, dataset_seed: int = 888):
    datafolder = dataset_folder / 'cifar10'
    trainset = datasets.CIFAR10(root=datafolder, train=True, download=False)
    testset = datasets.CIFAR10(root=datafolder, train=False, download=False)

    CLASS1, CLASS2 = classes

    train_size = num_data

    for partition in ['train', 'test']:
        partition_set = trainset if partition == 'train' else testset
        
        idx1 = [i for i, target in enumerate(partition_set.targets) if target == CLASS1]
        idx2 = [i for i, target in enumerate(partition_set.targets) if target == CLASS2]

        if partition == 'train':
            idx1 = idx1[:train_size // 2]
            idx2 = idx2[:train_size // 2]

        idx = idx1 + idx2
        idx.sort()

        partition_data = partition_set.data[idx]
        partition_target = np.array(partition_set.targets)[idx]
        partition_target = np.array([-1 if target == CLASS2 else 1 for target in partition_target])

        X = T.tensor(partition_data)
        Y = T.tensor(partition_target)

        # Normalize the images
        X = X / 255.0
        X = X.float()

        X = X - T.tensor([0.4914, 0.4822, 0.4465])
        X = X / T.tensor([0.2023, 0.1994, 0.2010])

        X = rearrange(X, 'b w h c -> b c w h')

        X = X.detach()

        # Now Y
        Y = Y.float()
        Y = Y.detach()

        if partition == 'train':
            X_train = X
            Y_train = Y
        else:
            X_test = X
            Y_test = Y
        
    return X_train, Y_train, X_test, Y_test


def prepare_cifar10(dataset_folder: Path, 
                    num_data: int, 
                    dataset_seed: int = 888,
                    loss_type: str = 'mse'
                    ):
    datafolder = dataset_folder / 'cifar10'
    trainset = datasets.CIFAR10(root=datafolder, train=True, download=False)
    testset = datasets.CIFAR10(root=datafolder, train=False, download=False)

    train_size = num_data

    for partition in ['train', 'test']:
        partition_set = trainset if partition == 'train' else testset
        
        if partition == 'train':
            rng = np.random.default_rng(dataset_seed)
            idx = rng.choice(len(partition_set), train_size, replace=False)

        else:
            idx = list(range(len(partition_set)))

        partition_data = partition_set.data[idx]
        partition_target = np.array(partition_set.targets)[idx]

        X = T.tensor(partition_data)
        Y = T.tensor(partition_target)

        # Normalize the images
        X = X / 255.0
        X = X.float()

        X = X - T.tensor([0.4914, 0.4822, 0.4465])
        X = X / T.tensor([0.2023, 0.1994, 0.2010])

        X = rearrange(X, 'b w h c -> b c w h')

        X = X.detach().float()

        # Now Y
        if loss_type == 'mse':
            Y = F.one_hot(Y, num_classes=10).float()
        else:
            Y = Y.long()


        if partition == 'train':
            X_train = X
            Y_train = Y
        else:
            X_test = X
            Y_test = Y
        
    return X_train, Y_train, X_test, Y_test

def prepare_fmnist(dataset_folder: Path, 
                            num_data: int, 
                            dataset_seed: int = 888,
                            loss_type: str = 'mse'
                            ):
    datafolder = dataset_folder / 'fmnist'
    trainset = datasets.FashionMNIST(root=datafolder, train=True, download=False)
    testset = datasets.FashionMNIST(root=datafolder, train=False, download=False)

    train_size = num_data

    for partition in ['train', 'test']:
        partition_set = trainset if partition == 'train' else testset
        
        if partition == 'train':
            rng = np.random.default_rng(dataset_seed)
            idx = rng.choice(len(partition_set), train_size, replace=False)
        else:
            idx = list(range(len(partition_set)))

        partition_data = partition_set.data[idx]
        partition_target = np.array(partition_set.targets)[idx]

        X = partition_data.unsqueeze(1)  # Add channel dimension
        Y = T.tensor(partition_target)

        # Normalize the images
        X = X / 255.0
        X = X.float()

        # Standard normalization for Fashion-MNIST
        X = X - 0.2860
        X = X / 0.3530

        X = X.detach().float()

        # Now Y
        if loss_type == 'mse':
            Y = F.one_hot(Y, num_classes=10).float()
        else:
            Y = Y.long()

        if partition == 'train':
            X_train = X
            Y_train = Y
        else:
            X_test = X
            Y_test = Y
        
    return X_train, Y_train, X_test, Y_test

def prepare_cifar100(num_data: int):
    pass

def prepare_svhn(dataset_folder: Path, 
                    num_data: int, 
                    dataset_seed: int = 888,
                    loss_type: str = 'mse'
                    ):
    import scipy.io as sio
    
    datafolder = dataset_folder / 'svhn'

    svhn_train_path = datafolder / 'train_32x32.mat'
    svhn_test_path = datafolder / 'test_32x32.mat'

    # Load training data
    train_data = sio.loadmat(svhn_train_path)
    X_svhn_train = np.transpose(train_data['X'], (3, 0, 1, 2))  # Convert to (N, H, W, C)
    Y_svhn_train = train_data['y'].reshape(-1)
    # SVHN labels are from 1-10, with 10 representing 0. Convert to 0-9
    Y_svhn_train[Y_svhn_train == 10] = 0

    # Load test data
    test_data = sio.loadmat(svhn_test_path)
    X_svhn_test = np.transpose(test_data['X'], (3, 0, 1, 2))
    Y_svhn_test = test_data['y'].reshape(-1)
    Y_svhn_test[Y_svhn_test == 10] = 0

    # Convert to torch tensors
    X_svhn_train = torch.from_numpy(X_svhn_train).float() / 255.0  # Normalize to [0, 1]
    Y_svhn_train = torch.from_numpy(Y_svhn_train).long()
    X_svhn_test = torch.from_numpy(X_svhn_test).float() / 255.0
    Y_svhn_test = torch.from_numpy(Y_svhn_test).long()


    for partition in ['train', 'test']:
        if partition == 'train':
            X = X_svhn_train
            Y = Y_svhn_train
            # If num_data is specified, limit the training data
            if num_data > 0 and num_data <= len(X_svhn_train):
                rng = np.random.default_rng(dataset_seed)
                idx = rng.choice(len(X_svhn_train), num_data, replace=False)
                X = X[idx]
                Y = Y[idx]
        else:
            X = X_svhn_test[:10_000]
            Y = Y_svhn_test[:10_000]
        


        # Normalize the images
        X = X # THE IMAGES ARE ALREADY NORMALIZED
        X = X.float()

        # Normalize using precomputed statistics
        X = X - T.tensor([0.4377, 0.4438, 0.4728])
        X = X / T.tensor([0.1980, 0.2010, 0.1970])

        X = rearrange(X, 'b w h c -> b c w h')

        X = X.detach().float()

        # Now Y
        if loss_type == 'mse':
            Y = F.one_hot(Y, num_classes=10).float()
        else:
            Y = Y.long()


        if partition == 'train':
            X_train = X
            Y_train = Y
        else:
            X_test = X
            Y_test = Y
    
    return X_train, Y_train, X_test, Y_test



def prepare_cifar10_ez(dataset_folder: Path, num_data: int, dataset_seed: int = 888, loss_type: str = 'mse'):
    # Define the file paths
    cifar_folder = dataset_folder / 'cifar10_ez'
    train_x_path = cifar_folder / 'X_train_pulled.npy'
    train_y_path = cifar_folder / 'Y_train.npy'
    test_x_path = cifar_folder / 'X_test_pulled.npy'
    test_y_path = cifar_folder / 'Y_test.npy'
    
    # Load the data
    X_train = torch.tensor(np.load(train_x_path)).float()
    Y_train = torch.tensor(np.load(train_y_path))
    X_test = torch.tensor(np.load(test_x_path)).float()
    Y_test = torch.tensor(np.load(test_y_path))
    
    # If num_data is specified, limit the training data
    if num_data > 0 and num_data <= len(X_train):
        rng = np.random.default_rng(dataset_seed)
        idx = rng.choice(len(X_train), num_data, replace=False)
        X_train = X_train[idx]
        Y_train = Y_train[idx]
    
    # Handle Y based on loss_type
    if loss_type == 'mse':
        Y_train = Y_train.float()
        Y_test = Y_test.float()
        #F.one_hot(Y_test.long(), num_classes=10).float()
    else:
        raise NotImplementedError("Cross-entropy loss is not supported for CIFAR-10 EZ dataset YET")
        Y_train = Y_train.long()
        Y_test = Y_test.long()
    
    return X_train, Y_train, X_test, Y_test


def load_batch(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read one ImageNet-32 pickle batch and return images (HWC uint8) & labels."""
    with open(path, "rb") as f:
        entry = pickle.load(f, encoding="bytes")

    data = entry["data"]          # (N, 3072) uint8
    labels = entry["labels"]

    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(labels, dtype=np.int64)
    return images, labels


def load_imagenet32(root: str, train: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Load all train batches or the validation split into memory."""
    if train:
        batches = sorted(f for f in os.listdir(root) if f.startswith("train_data_batch_"))
        if not batches:
            raise FileNotFoundError("No 'train_data_batch_*' files found in " + root)
    else:
        batches = ["val_data"]

    imgs_list, lbls_list = [], []
    for bname in batches:
        imgs, lbls = load_batch(os.path.join(root, bname))
        imgs_list.append(imgs)
        lbls_list.append(lbls)

    return np.concatenate(imgs_list), np.concatenate(lbls_list)


class ImageNet32(Dataset):  # type: ignore[misc]
    """PyTorch-friendly wrapper (optional)."""

    def __init__(self, root: str, train: bool = True, transform=None, target_transform=None):
        self.images, self.labels = load_imagenet32(root, train)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):  # type: ignore[override]
        return self.images.shape[0]

    def __getitem__(self, idx):  # type: ignore[override]
        img, lbl = self.images[idx], int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        elif torch and not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if self.target_transform:
            lbl = self.target_transform(lbl)
        return img, lbl


def prepare_imagenet32(dataset_folder: Path, num_data: int, dataset_seed: int = 888, loss_type: str = 'mse'):
    """Prepare ImageNet32 dataset for training."""
    datafolder = dataset_folder / 'imagenet32'
    
    # Load train and test data
    train_images, train_labels = load_imagenet32(str(datafolder), train=True)
    test_images, test_labels = load_imagenet32(str(datafolder), train=False)
    
    train_size = num_data

    for partition in ['train', 'test']:
        if partition == 'train':
            partition_data = train_images
            partition_target = train_labels
            
            if num_data > 0 and num_data < len(partition_data):
                rng = np.random.default_rng(dataset_seed)
                idx = rng.choice(len(partition_data), train_size, replace=False)
                partition_data = partition_data[idx]
                partition_target = partition_target[idx]
        else:
            partition_data = test_images
            partition_target = test_labels

        X = T.tensor(partition_data)
        Y = T.tensor(partition_target)

        # Normalize the images
        X = X / 255.0
        X = X.float()

        # ImageNet normalization values
        X = X - T.tensor([0.485, 0.456, 0.406])
        X = X / T.tensor([0.229, 0.224, 0.225])

        X = rearrange(X, 'b w h c -> b c w h')

        X = X.detach().float()

        # Now Y
        Y = Y - 1 # ImageNet labels are from 1 to 1000, not 0 to 999
        if loss_type == 'mse':
            Y = F.one_hot(Y, num_classes=1000).float()  # ImageNet has 1000 classes
        else:
            Y = Y.long()

        if partition == 'train':
            X_train = X
            Y_train = Y
        else:
            X_test = X
            Y_test = Y
        
    return X_train, Y_train, X_test, Y_test




def prepare_shakespeare(dataset_folder: Path, num_data: int, seq_len: int = 128, dataset_seed: int = 888):
    import urllib.request
    shakes_dir = dataset_folder / 'shakespeare'
    shakes_dir.mkdir(parents=True, exist_ok=True)
    txt_path = shakes_dir / 'input.txt'
    if not txt_path.exists():
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        urllib.request.urlretrieve(url, txt_path)

    text = txt_path.read_text()

    # Build char vocab (sorted for determinism)
    chars = sorted(set(text))
    assert len(chars) == 65, f"Expected 65 unique chars, got {len(chars)}"
    stoi = {c: i for i, c in enumerate(chars)}

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    # 90/10 train/test split
    n = len(data)
    n_train = int(0.9 * n)
    train_data = data[:n_train]
    test_data = data[n_train:]

    def make_windows(d):
        # Non-overlapping windows of length seq_len
        n_seq = len(d) // (seq_len + 1)
        X = torch.stack([d[i * seq_len: i * seq_len + seq_len] for i in range(n_seq)])
        Y = torch.stack([d[i * seq_len + 1: i * seq_len + seq_len + 1] for i in range(n_seq)])
        return X, Y

    X_train, Y_train = make_windows(train_data)
    X_test, Y_test = make_windows(test_data)

    # Subsample training set
    if num_data > 0 and num_data < len(X_train):
        rng = np.random.default_rng(dataset_seed)
        idx = rng.choice(len(X_train), num_data, replace=False)
        X_train = X_train[idx]
        Y_train = Y_train[idx]

    return X_train, Y_train, X_test, Y_test


def prepare_sst2(dataset_folder: Path, num_data: int, seq_len: int = 64, dataset_seed: int = 0):
    """Load SST-2 from HuggingFace, replicating Damian et al. (arXiv:2209.15594) exactly.

    Returns n=num_data class-balanced samples (num_data//2 negative, num_data//2 positive),
    tokenized with bert-base-uncased, padded/truncated to seq_len=64.
    Labels are float32 {-1, +1}. No test split (matches Damian).
    """
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_dataset("glue", "sst2")
    y = np.array(dataset["train"]["label"])
    n_per_class = num_data // 2
    neg_idx = np.argwhere(y == 0)[:n_per_class, 0]
    pos_idx = np.argwhere(y == 1)[:n_per_class, 0]
    idx = np.concatenate([neg_idx, pos_idx])
    labels = 2 * y[idx] - 1  # {0,1} -> {-1,+1}
    sentences = [dataset["train"][int(i)]["sentence"] for i in idx]
    input_ids = tokenizer(
        sentences, padding='max_length', truncation=True, max_length=seq_len
    )["input_ids"]
    X = torch.tensor(input_ids, dtype=torch.long)
    Y = torch.tensor(labels, dtype=torch.float32)
    return X, Y, None, None


def prepare_dataset(dataset: str, dataset_folder: Union[str, Path], num_data: int, classes: list, dataset_seed: int = 888,
                    loss_type: str = 'mse'
                    ):
    dataset_folder = Path(dataset_folder)
    train_size = num_data
    if dataset == 'cifar10_2cls':
        if loss_type == 'ce':
            raise NotImplementedError("Cross-entropy loss is not supported for 2-class CIFAR-10 dataset YET")
        return prepare_cifar10_2cls(dataset_folder, num_data, classes, dataset_seed=dataset_seed)
    if dataset == 'cifar10':
        return prepare_cifar10(dataset_folder, num_data, dataset_seed=dataset_seed, loss_type=loss_type)
    if dataset == 'cifar10_ez':
        return prepare_cifar10_ez(dataset_folder, num_data, dataset_seed=dataset_seed, loss_type=loss_type)
    if dataset == 'svhn':
        return prepare_svhn(dataset_folder, num_data, dataset_seed=dataset_seed, loss_type=loss_type)
    if dataset == 'fmnist':
        return prepare_fmnist(dataset_folder, num_data, dataset_seed=dataset_seed, loss_type=loss_type)
    if dataset == 'imagenet32':
        return prepare_imagenet32(dataset_folder, num_data, dataset_seed=dataset_seed, loss_type=loss_type)
    if dataset == 'shakespeare':
        return prepare_shakespeare(dataset_folder, num_data, seq_len=128, dataset_seed=dataset_seed)
    if dataset == 'sst2':
        return prepare_sst2(dataset_folder, num_data, seq_len=64, dataset_seed=dataset_seed)

