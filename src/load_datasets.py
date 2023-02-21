# pyright: reportMissingImports=true, reportUntypedBaseClass=false, reportGeneralTypeIssues=false
from typing import List, Union

import os
import copy
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
#from lacuna import Lacuna10, Lacuna100, Small_Lacuna10, Small_Binary_Lacuna10, Small_Lacuna5
from datasets.Small_CIFAR10 import Small_CIFAR10, Small_CIFAR5
from datasets.Small_MNIST import Small_MNIST, Small_Binary_MNIST
from datasets.TinyImageNet import TinyImageNet_pretrain, TinyImageNet_finetune, TinyImageNet_finetune5
from utils import seed_everything

def manual_seed(seed):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
_DATASETS = {}

def _add_dataset(dataset_fn):
    _DATASETS[dataset_fn.__name__] = dataset_fn
    return dataset_fn

def _get_mnist_transforms():
    transform_augment = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),])
    transform_test = transforms.Compose([
        transforms.ToTensor(),])
    transform_train = transform_augment

    return transform_train, transform_test

# def _get_lacuna_transforms(augment=True):
#     transform_augment = transforms.Compose([
#         transforms.RandomCrop(64, padding=4),
#         transforms.Resize(size=(32, 32)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.382, 0.420, 0.502), (0.276, 0.279, 0.302)),
#     ])
#     transform_test = transforms.Compose([
#         transforms.Resize(size=(32, 32)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.382, 0.420, 0.502), (0.276, 0.279, 0.302)),
#     ])
#     transform_train = transform_augment if augment else transform_test

#     return transform_train, transform_test

def _get_cifar_transforms():
    transform_augment = transforms.Compose([
        transforms.Pad(padding=4, fill=(125,123,113)),
        transforms.RandomCrop(32, padding=0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_train = transform_augment

    return transform_train, transform_test

def _get_imagenet_transforms():
    transform_augment = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.Resize(size=(32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_train = transform_augment

    return transform_train, transform_test

def _get_tinyimagenet_transforms():
    transform_augment = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_train = transform_augment

    return transform_train, transform_test

@_add_dataset   
def cifar10(root, transform_tr=True):
    transform_train, transform_test = _get_cifar_transforms()
    if not transform_tr:
        transform_train = transform_test
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    test_set  = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    return train_set, test_set

@_add_dataset   
def small_cifar5(root, transform_tr=True):
    transform_train, transform_test = _get_cifar_transforms()
    if not transform_tr:
        print('Debug: train_full set transforms acc. inference')
        transform_train = transform_test
    train_set = Small_CIFAR5(root=root, train=True, transform=transform_train)
    test_set  = Small_CIFAR5(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset   
def small_cifar10(root, transform_tr=True):
    transform_train, transform_test = _get_cifar_transforms()
    if not transform_tr:
        transform_train = transform_test
    train_set = Small_CIFAR10(root=root, train=True, transform=transform_train)
    test_set  = Small_CIFAR10(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset   
def cifar100(root, transform_tr=True):
    transform_train, transform_test = _get_cifar_transforms()
    if not transform_tr:
        transform_train = transform_test
    train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
    test_set  = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)
    return train_set, test_set

@_add_dataset
def mnist(root, transform_tr=True):
    transform_train, transform_test = _get_mnist_transforms()
    if not transform_tr:
        transform_train = transform_test
    train_set = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform_test)
    return train_set, test_set

@_add_dataset
def small_mnist(root, transform_tr=True):
    transform_train, transform_test = _get_mnist_transforms()
    if not transform_tr:
        transform_train = transform_test
    train_set = Small_MNIST(root=root, train=True, transform=transform_train)
    test_set = Small_MNIST(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def small_binary_mnist(root, transform_tr=True):
    transform_train, transform_test = _get_mnist_transforms()
    if not transform_tr:
        transform_train = transform_test
    train_set = Small_Binary_MNIST(root=root, train=True, transform=transform_train)
    test_set = Small_Binary_MNIST(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def tinyimagenet_pretrain(root, transform_tr=True):
    transform_train, transform_test = _get_imagenet_transforms()
    if not transform_tr:
        transform_train = transform_test
    train_set = TinyImageNet_pretrain(root=root, train=True, transform=transform_train)
    test_set = TinyImageNet_pretrain(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def tinyimagenet_finetune(root, transform_tr=True):
    transform_train, transform_test = _get_imagenet_transforms()
    if not transform_tr:
        transform_train = transform_test
    train_set = TinyImageNet_finetune(root=root, train=True, transform=transform_train)
    test_set = TinyImageNet_finetune(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def tinyimagenet_finetune5(root, transform_tr=True):
    transform_train, transform_test = _get_imagenet_transforms()
    if not transform_tr:
        transform_train = transform_test
    train_set = TinyImageNet_finetune5(root=root, train=True, transform=transform_train)
    test_set = TinyImageNet_finetune5(root=root, train=False, transform=transform_test)
    return train_set, test_set


def replace_indexes(logger, dataset: torch.utils.data.Dataset, indexes: Union[List[int], np.ndarray], only_mark: bool = False):
    if not only_mark:
        ''' Golatkar's replacement
        rng = np.random.RandomState(seed)
        new_indexes = rng.choice(list(set(range(len(dataset))) - set(indexes)), size=len(indexes))
        dataset.data[indexes] = dataset.data[new_indexes]
        dataset.targets[indexes] = dataset.targets[new_indexes]
        '''
        old_size = len(dataset.data)
        retain_indexes = list(set(range(len(dataset))) - set(indexes))
        logger.debug(f'Removed targets: {dataset.targets[indexes]}')
        dataset.data = dataset.data[retain_indexes]
        dataset.targets = dataset.targets[retain_indexes]
        new_size = len(dataset.data)
        logger.debug(f'Oldsize: {old_size}\t Newsize:{new_size}')
        return dataset
    else:
        # Notice the -1 to make class 0 work
        logger.debug(f'Marking forget set targets: {dataset.targets[indexes]}')
        dataset.targets[indexes] = - dataset.targets[indexes] - 1
        return dataset


def replace_class(logger, dataset: torch.utils.data.Dataset, class_to_replace: int, num_indexes_to_replace: int = None,
                  seed: int = 0, only_mark: bool = False):
    indexes = np.flatnonzero(np.array(dataset.targets) == class_to_replace)
    if num_indexes_to_replace is not None:
        assert num_indexes_to_replace <= len(
            indexes), f"Want to replace {num_indexes_to_replace} indexes but only {len(indexes)} samples in dataset"
        rng = np.random.RandomState(seed)
        indexes = rng.choice(indexes, size=num_indexes_to_replace, replace=False)
        logger.info(f"Replacing indexes {indexes}")
    return replace_indexes(logger, dataset, indexes, only_mark)

'''
Added by Shash: Confusion creator in data loader
Assumes confusion is a n_C x n_C matrix s.t. 
C_{i, j} = relabel C_{i, j} samples of class i as j
Will only work for small datasets ofc, should be optimized further
'''

#LEFT: Copy, Optimize impl., Quantification of confusion
def add_confusion(dataset: torch.utils.data.Dataset, confusion: List[List[int]], conf_copy: bool):
    num_classes = len(confusion)
    confset = copy.deepcopy(dataset)
    confused_indices = []
    for i in range(num_classes):
        class_i_indexes = np.flatnonzero(np.array(dataset.targets) == i)
        last_rep = 0
        for j in range(num_classes):
            if i==j:
                continue
            for itr in range(confusion[i][j]): #Replace first C_{i, j} starting from last_rep indexes
                idx = class_i_indexes[last_rep + itr]
                confset.targets[idx] = j
                confused_indices.append(idx)
            last_rep += confusion[i][j]

    return confset, confused_indices, dataset

def get_loaders(logger, dataset_name, validsplit = 0.2, confusion: List[List[int]] = None, confusion_copy=False, class_to_replace: int = None, 
                num_indexes_to_replace: int = None, indexes_to_replace: List[int] = None, seed: int = 1, 
                only_mark: bool = False, root: str = None, batch_size=128, shuffle=True, **dataset_kwargs):
    '''

    :param dataset_name: Name of dataset to use
    :param class_to_replace: If not None, specifies which class to replace completely or partially
    :param num_indexes_to_replace: If None, all samples from `class_to_replace` are replaced. Else, only replace
                                   `num_indexes_to_replace` samples
    :param indexes_to_replace: If not None, denotes the indexes of samples to replace. Only one of class_to_replace and
                               indexes_to_replace can be specidied.
    :param seed: Random seed to sample the samples to replace and to initialize the data loaders so that they sample
                 always in the same order
    :param root: Root directory to initialize the dataset
    :param batch_size: Batch size of data loader
    :param shuffle: Whether train data should be randomly shuffled when loading (test data are never shuffled)
    :param dataset_kwargs: Extra arguments to pass to the dataset init.
    :return: The train_loader and test_loader
    '''
    seed_everything(seed)

    if root is None:
        root = os.path.expanduser('~/data')
    train_set, test_set = _DATASETS[dataset_name](root, **dataset_kwargs)
    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)
    
    valid_set = copy.deepcopy(train_set)
    rng = np.random.RandomState(seed)
    
    valid_idx=[]
    for i in range(max(train_set.targets) + 1):
        class_idx = np.where(train_set.targets==i)[0]
        valid_idx.append(rng.choice(class_idx,int(validsplit*len(class_idx)),replace=False))
    valid_idx = np.hstack(valid_idx)
    
    train_idx = list(set(range(len(train_set)))-set(valid_idx))
    
    train_set_copy = copy.deepcopy(train_set)

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]
        
    '''Modified here by Shash'''
    confused_indices = None
    if (class_to_replace is not None) + (indexes_to_replace is not None) + (confusion is not None) >= 2:
        raise ValueError("Only one of `class_to_replace`, `confusion` and `indexes_to_replace` can be specified")
    if confusion is not None:
        train_set, confused_indices, og_train_set = add_confusion(train_set, confusion, confusion_copy)
    if class_to_replace is not None:
        logger.info(f'C_f: {class_to_replace}\t N_f: {num_indexes_to_replace}\t only_mark: {only_mark}\t train_size: {len(train_set.data)}')
        if class_to_replace == -1:
            rng = np.random.RandomState(seed-1)
            indexes = rng.choice(range(len(train_set)), size=num_indexes_to_replace, replace=False)
            train_set = replace_indexes(logger, train_set, indexes, only_mark)
        else:
            train_set = replace_class(logger, train_set, class_to_replace, num_indexes_to_replace=num_indexes_to_replace,
                                     seed=seed-1,only_mark=only_mark)
        #train_set could be both 'dataset' and 'subset'.
    elif indexes_to_replace is not None:
        train_set = replace_indexes(logger, dataset=train_set, indexes=indexes_to_replace, only_mark=only_mark)

    loader_args = {'num_workers': 0, 'pin_memory': False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,
                                               worker_init_fn=_init_fn if seed is not None else None, **loader_args)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                                              worker_init_fn=_init_fn if seed is not None else None, **loader_args)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              worker_init_fn=_init_fn if seed is not None else None, **loader_args)

    return train_loader, valid_loader, test_loader, confused_indices

