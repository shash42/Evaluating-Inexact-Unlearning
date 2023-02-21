# pyright: reportGeneralTypeIssues=false
import os
import random
import logging
import numpy as np
import torch
import errno
import copy
import models
import argparse

def load_pretrained(model, model_name, pretrain_path):
    classifier_name = 'linear.'
    if model_name=='allcnn':   classifier_name='classifier.'
    elif 'resnet' in model_name:    classifier_name='linear.'
    if pretrain_path is not None:
        checkpoint = torch.load(pretrain_path, map_location=torch.device('cuda'))
        state = {k: v for k, v in checkpoint['state_dict'].items() if not k.startswith(classifier_name)}
        incompatible_keys = model.load_state_dict(state, strict=False)
        # print(f'State: {state}\nMissing: {incompatible_keys.missing_keys}\nUnexpected: {incompatible_keys.unexpected_keys}')
        assert all([k.startswith(classifier_name) for k in incompatible_keys.missing_keys])
    return model

#[TODO]: Combine/Organize/Re-order functions here
def dataset_subset(dataset, idxes):
    dataset.data = dataset.data[idxes]
    dataset.targets = dataset.targets[idxes]
    return dataset


#[TODO:] Ensure this is deterministic
def split_loader(loader, ratio, batch_sz1=1, batch_sz2=1):
    '''
    Given a dataloader, splits it's dataset based on the given ratio
    Returns 2 loaders with sizes acc. to given ratio
    No shuffling is assumed elsewhere.
    '''
    idxs1 = np.random.choice(len(loader.dataset), int(ratio*len(loader.dataset)), replace=False)
    idxs2 = list(set(range(len(loader.dataset)))-set(idxs1))
    dataset1 = dataset_subset(copy.deepcopy(loader.dataset), idxs1)
    dataset2 = dataset_subset(copy.deepcopy(loader.dataset), idxs2)
    loader1 = torch.utils.data.DataLoader(dataset1, batch_size=batch_sz1, shuffle=False)
    loader2 = torch.utils.data.DataLoader(dataset2, batch_size=batch_sz2, shuffle=False)
    return loader1, loader2, idxs1, idxs2

#[TODO:] Ensure this is deterministic
def subset_loader(loader, size, idxs=None, batch_sz=1):
    '''
    Return new dataloader with dataset as a subset of 'size' of passed 'loader'
    Pass idxs if they are to be specified, otherwise if None chooses randomly.
    No shuffling is assumed elsewhere.
    '''
    if idxs is None: #Take random indices
        idxs = np.random.choice(len(loader.dataset), size, replace=False)
    dataset = dataset_subset(copy.deepcopy(loader.dataset), idxs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_sz, shuffle=False)
    return dataloader, idxs

def mislabel_loader(loader, label_og, label_new):
    for i, y in enumerate(loader.dataset.targets):
        if y == label_og:
            loader.dataset.targets[i] = label_new
    return loader

def loader_exch2_classes(loader, exch):
    loader = mislabel_loader(loader, exch[0], -exch[1])
    loader = mislabel_loader(loader, exch[1], -exch[0]) #this and next step can combined, i.e. directly exch[1] -> exch[0]
    loader = mislabel_loader(loader, -exch[0], exch[0])
    loader = mislabel_loader(loader, -exch[1], exch[1])
    return loader

def pdb():
    import pdb
    pdb.set_trace

def trainable_params_(m):
    return [p for p in m.parameters() if p.requires_grad]
    
def parameter_count(logger, model):
    count=0
    for p in model.parameters():
        count+=np.prod(np.array(list(p.shape)))
    logger.debug(f'Total Number of Parameters: {count}')

def print_param_shape(logger, model):
    for k,p in model.named_parameters():
        logger.debug(k,p.shape)

def distance(logger, model, model0):
    distance=0
    normalization=0
    for (k, p), (k0, p0) in zip(model.named_parameters(), model0.named_parameters()):
        space='  ' if 'bias' in k else ''
        current_dist=(p.data0-p0.data0).pow(2).sum().item()
        current_norm=p.data0.pow(2).sum().item()
        distance+=current_dist
        normalization+=current_norm
    logger.info(f'Distance: {np.sqrt(distance)}')
    logger.info(f'Normalized Distance: {1.0*np.sqrt(distance/normalization)}')
    return 1.0*np.sqrt(distance/normalization)

#Called ntk_init in original notebook
def get_model_init(device, resume, arch, num_classes, filters, seed=1):
    seed_everything(seed)
    model_init = models.get_model(arch, num_classes=num_classes, filters_percentage=filters).to(device)
    model_init.load_state_dict(torch.load(resume))
    return model_init

#[TODO:] Do we need this wrapper?!
def get_loader_from_dataset(dataset, batch_size, seed=1, shuffle=True):
    seed_everything(seed)
    '''Unused lines in Golatkar notebook
    loader_args = {'num_workers': 0, 'pin_memory': False}
    def _init_fn(worker_id):
        np.random.seed(int(seed))
    '''
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,num_workers=0,pin_memory=True,shuffle=shuffle)


def test_loader_split(test_loader_full, seed, batch_size, exch_classes=None, forget_class=None):
    test_dataset_full = copy.deepcopy(test_loader_full.dataset)
    test_dataset_r = copy.deepcopy(test_loader_full.dataset)
    test_dataset_f = copy.deepcopy(test_loader_full.dataset)
    
    test_idxes_f, test_idxes_r = None, None
    if exch_classes is not None:
        test_idxes_f = [target in exch_classes for target in test_dataset_full.targets]
        test_idxes_r = [target not in exch_classes for target in test_dataset_full.targets]
    elif forget_class is not None and forget_class != -1:
        test_idxes_f = test_dataset_full.targets == forget_class
        test_idxes_r = test_dataset_full.targets != forget_class
    else:
        return None, None
    
    test_dataset_f.data, test_dataset_f.targets = test_dataset_f.data[test_idxes_f], test_dataset_f.targets[test_idxes_f]
    test_dataset_r.data, test_dataset_r.targets = test_dataset_r.data[test_idxes_r], test_dataset_r.targets[test_idxes_r]
    assert(len(test_dataset_f) + len(test_dataset_r) == len(test_loader_full.dataset))
    test_loader_f = get_loader_from_dataset(test_dataset_f, batch_size, seed=seed, shuffle=True)
    test_loader_r = get_loader_from_dataset(test_dataset_r, batch_size, seed=seed, shuffle=True)
    
    return test_loader_r, test_loader_f

#[TODO]: How to check if this gives same data in forget_dataset (correct labels), forget_dataset_og (potentially mislabelled)?!
def get_forget_retain_loader(marked_loader, train_loader_full, seed, batch_size):
    forget_dataset = copy.deepcopy(marked_loader.dataset)
    marked = forget_dataset.targets < 0
    
    print(f'Forget (marked) dataset labels: {forget_dataset.targets[marked]}')
    forget_dataset.data = forget_dataset.data[marked]
    forget_dataset.targets = - forget_dataset.targets[marked] - 1
    forget_loader = get_loader_from_dataset(forget_dataset, batch_size, seed=seed, shuffle=True)

    forget_dataset_og = copy.deepcopy(train_loader_full.dataset)
    forget_dataset_og.data = forget_dataset_og.data[marked]
    forget_dataset_og.targets = forget_dataset_og.targets[marked]
    forget_loader_og = get_loader_from_dataset(forget_dataset_og, batch_size, seed=seed, shuffle=True)

    retain_dataset = copy.deepcopy(marked_loader.dataset)
    marked = retain_dataset.targets >= 0
    retain_dataset.data = retain_dataset.data[marked]
    retain_dataset.targets = retain_dataset.targets[marked]
    retain_loader = get_loader_from_dataset(retain_dataset, batch_size, seed=seed, shuffle=True)

    assert(len(forget_dataset) + len(retain_dataset) == len(train_loader_full.dataset))
    return forget_loader, retain_loader, forget_loader_og

def print_final_times(logger, t_pre, t, t0, t_ntk, t_f, t_ntkf, t_ft, t_r):
    logger.info(f'\nPrecompute\t{t_pre}\n Original\t{t}\n Retrain\t{t0}\n NTK\t{t_ntk}\n'
        f' Fisher\t{t_f}\n NTKFisher\t{t_ntkf}\n Finetune\t{t_ft}\n RetrainLast\t{t_r}\n')

'''Shash: Changed to avoid duplicate logging - https://stackoverflow.com/a/7175288'''
loggers = {}
def get_logger(folder, logname):
    global loggers
    if loggers.get(f'{folder}-{logname}'):
        return loggers.get(f'{folder}-{logname}')
    else:
        logger = logging.getLogger(f'{folder}-{logname}')
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
        # file logger
        if not os.path.isdir(folder):
            os.mkdir(folder)
        fh = logging.FileHandler(os.path.join(folder, f'{logname}.log'), mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        # console logger
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        loggers[f'{folder}-{logname}'] = logger
    return logger

def seed_everything(seed):
    '''
    Fixes the class-to-task assignments and most other sources of randomness, except CUDA training aspects.
    '''
    # Avoid all sorts of randomness for better replication
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True # An exemption for speed :P

def save_model(filepath, model):
    '''
    Used for saving the pretrained model, not for intermediate breaks in running the code.
    '''
    state = {'state_dict': model.state_dict()}
    torch.save(state, filepath)


def load_model(filepath, model, logger):
    '''
    Used for loading the pretrained model, not for intermediate breaks in running the code.
    '''
    logger.info("=> loading checkpoint '{}'".format(filepath))
    assert(os.path.isfile(filepath))
    checkpoint = torch.load(filepath, map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint['state_dict'])
    return model

def cutmix_data(x, y, alpha=1.0, cutmix_prob=0.5):
    assert(alpha > 0)
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    if torch.cuda.is_available():
        index = index.cuda()

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
    
class AverageMeter:
    # Sourced from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    def __init__(self):
        self.reset()
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum*1.0 / self.count*1.0

def get_error(output, target):
    if output.shape[1]>1:
        pred = output.argmax(dim=1, keepdim=True)
        return 1. - pred.eq(target.view_as(pred)).float().mean().item()
    else:
        pred = output.clone()
        pred[pred>0]=1
        pred[pred<=0]=-1
        return 1 - pred.eq(target.view_as(pred)).float().mean().item()
        
def set_batchnorm_mode(model, train=True):
    if isinstance(model, torch.nn.BatchNorm1d) or isinstance(model, torch.nn.BatchNorm2d):
        if train:
            model.train()
        else:
            model.eval()
    for l in model.children():
        set_batchnorm_mode(l, train=train)

def print_model_params(model, path):
    with open(path, 'w') as f:
        for name, param in model.named_parameters():
            print(name, param, file=f)
        
def mkdir(directory):
    '''Make directory and all parents, if needed.
    Does not raise and error if directory already exists.
    '''

    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
            
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def argparse2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')