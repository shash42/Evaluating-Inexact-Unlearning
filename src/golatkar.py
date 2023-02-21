#Methods of Golatkar et al. - NTK, Fisher, NTK+Fisher, Finetune
# pyright: reportMissingImports=true, reportUntypedBaseClass=false, reportGeneralTypeIssues=true
import torch
import torch.nn as nn
import copy
import numpy as np
import os
from tqdm import tqdm
import random
import json
import models
from collections import OrderedDict, defaultdict
import torch.nn.functional as F

def manual_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def vectorize_params(model):
    param = []
    for p in model.parameters():
        param.append(p.data.view(-1).cpu().numpy())
    return np.concatenate(param)

def log_metrics(split, metrics, epoch, **kwargs):
    print(f'[{epoch}] {split} metrics:' + json.dumps(metrics.avg))

def get_error(output, target):
    if output.shape[1]>1:
        pred = output.argmax(dim=1, keepdim=True)
        return 1. - pred.eq(target.view_as(pred)).float().mean().item()
    else:
        pred = output.clone()
        pred[pred>0]=1
        pred[pred<=0]=-1
        return 1 - pred.eq(target.view_as(pred)).float().mean().item()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = defaultdict(int)
        self.avg = defaultdict(float)
        self.sum = defaultdict(int)
        self.count = defaultdict(int)

    def update(self, n=1, **val):
        for k in val:
            self.val[k] = val[k]
            self.sum[k] += val[k] * n
            self.count[k] += n
            self.avg[k] = self.sum[k] / self.count[k]

def ntk_init(device, resume, arch, num_classes, filters, seed=1):
    manual_seed(seed)
    model_init = models.get_model(arch, num_classes=num_classes, filters_percentage=filters).to(device)
    model_init.load_state_dict(torch.load(resume))
    for p in model_init.parameters():
        p.data0 = p.data.clone()
    return model_init

def golatkar_precomputation(model, model0, retain_loader, forget_loader, seed, device, dataset, lossfn, wt_decay,
                            init_checkpoint, arch, num_classes, num_total, num_to_retain, filters, pathprefix):
    #NTK/Fisher Stuff Prep [Jacobians and Hessians]
    model_init = create_jachess(model, retain_loader, forget_loader, seed, device, dataset, lossfn, wt_decay, 
                                init_checkpoint, arch, num_classes, filters, num_total, num_to_retain, pathprefix)
    scale_ratio, delta_w, delta_w_actual, w_retain = scrubbing_direction(model, model0, pathprefix)
    print(f'Debug: delta_w_actual: {delta_w_actual} | delta_w: {delta_w}')
    predicted_scale = trapezium_trick(model, model_init, delta_w, w_retain)
    nip=NIP(delta_w_actual,delta_w)
    return model_init, scale_ratio, delta_w, delta_w_actual, w_retain, predicted_scale

def delta_w_utils(model_init, dataloader, dataset, lossfn, num_classes, name='complete'):
    model_init.eval()
    dataloader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=1, shuffle=False)
    G_list = []
    f0_minus_y = []
    for idx, batch in enumerate(dataloader):#(tqdm(dataloader,leave=False)):
        batch = [tensor.to(next(model_init.parameters()).device) for tensor in batch]
        input, target = batch
        if 'mnist' in dataset:
            input = input.view(input.shape[0],-1)
        target = target.cpu().detach().numpy()
        output = model_init(input)
        G_sample=[]
        for cls in range(num_classes):
            grads = torch.autograd.grad(output[0,cls],model_init.parameters(),retain_graph=True)
            grads = np.concatenate([g.view(-1).cpu().numpy() for g in grads])
            G_sample.append(grads)
            G_list.append(grads)
        f0_y_update = None
        if lossfn=='mse':
            p = output.cpu().detach().numpy().transpose()
            #loss_hess = np.eye(len(p))
            target = 2*target-1
            f0_y_update = p-target
        elif lossfn=='ce':
            p = torch.nn.functional.softmax(output,dim=1).cpu().detach().numpy().transpose()
            p[target]-=1
            f0_y_update = copy.deepcopy(p)
        f0_minus_y.append(f0_y_update)
    return np.stack(G_list).transpose(),np.vstack(f0_minus_y)


def create_jachess(model, retain_loader, forget_loader, seed, device, dataset, lossfn, wt_decay, init_checkpoint, arch, 
                    num_classes, filters, num_total, num_to_retain, prefixpath):
    if not os.path.exists(prefixpath):
        os.mkdir(prefixpath)
    
    model_init = ntk_init(device, init_checkpoint, arch, num_classes, filters, seed)
    G_r,f0_minus_y_r = delta_w_utils(copy.deepcopy(model),retain_loader, dataset, lossfn, num_classes, 'complete') 

    np.save(f"{prefixpath}/G_r.npy",G_r)
    np.save(f"{prefixpath}/f0_minus_y_r.npy",f0_minus_y_r)
    del G_r, f0_minus_y_r

    model_init = ntk_init(device, init_checkpoint, arch, num_classes, filters, seed)
    G_f,f0_minus_y_f = delta_w_utils(copy.deepcopy(model),forget_loader, dataset, lossfn, num_classes, 'retain') 

    np.save(f"{prefixpath}/G_f.npy",G_f)
    np.save(f"{prefixpath}/f0_minus_y_f.npy",f0_minus_y_f)
    del G_f, f0_minus_y_f

    G_r = np.load(f"{prefixpath}/G_r.npy")
    G_f = np.load(f"{prefixpath}/G_f.npy")
    G = np.concatenate([G_r,G_f],axis=1)

    np.save(f"{prefixpath}/G.npy",G)
    del G, G_f, G_r

    f0_minus_y_r = np.load(f"{prefixpath}/f0_minus_y_r.npy")
    f0_minus_y_f = np.load(f"{prefixpath}/f0_minus_y_f.npy")
    f0_minus_y = np.concatenate([f0_minus_y_r,f0_minus_y_f])

    np.save(f"{prefixpath}/f0_minus_y.npy",f0_minus_y)
    del f0_minus_y, f0_minus_y_r, f0_minus_y_f

    # w_lin(D)
    G = np.load(f"{prefixpath}/G.npy")
    theta = G.transpose().dot(G) + num_total*wt_decay*np.eye(G.shape[1])
    del G

    theta_inv = np.linalg.inv(theta)

    np.save(f"{prefixpath}/theta.npy",theta)
    del theta

    G = np.load(f"{prefixpath}/G.npy")
    f0_minus_y = np.load(f"{prefixpath}/f0_minus_y.npy")
    w_complete = -G.dot(theta_inv.dot(f0_minus_y))

    np.save(f"{prefixpath}/theta_inv.npy",theta_inv)
    np.save(f"{prefixpath}/w_complete.npy",w_complete)
    del G, f0_minus_y, theta_inv, w_complete

    #w_lin(D_r)
    G_r = np.load(f"{prefixpath}/G_r.npy")
    theta_r = G_r.transpose().dot(G_r) + num_to_retain*wt_decay*np.eye(G_r.shape[1])
    del G_r

    theta_r_inv = np.linalg.inv(theta_r)
    np.save(f"{prefixpath}/theta_r.npy",theta_r)
    del theta_r

    G_r = np.load(f"{prefixpath}/G_r.npy")
    f0_minus_y_r = np.load(f"{prefixpath}/f0_minus_y_r.npy")
    w_retain = -G_r.dot(theta_r_inv.dot(f0_minus_y_r))

    np.save(f"{prefixpath}/theta_r_inv.npy",theta_r_inv)
    np.save(f"{prefixpath}/w_retain.npy",w_retain)
    del G_r, f0_minus_y_r, theta_r_inv, w_retain  

    return model_init


def scrubbing_direction(model, model0, prefixpath):
    w_complete = np.load(f"{prefixpath}/w_complete.npy")
    w_retain = np.load(f"{prefixpath}/w_retain.npy")
    delta_w = (w_retain-w_complete).squeeze()
    
    #Actual change in weights
    delta_w_actual = vectorize_params(model0)-vectorize_params(model)
    print(f'Actual Norm-: {np.linalg.norm(delta_w_actual)}')
    print(f'Predtn Norm-: {np.linalg.norm(delta_w)}')
    scale_ratio = np.linalg.norm(delta_w_actual)/np.linalg.norm(delta_w)
    print('Actual Scale: {}'.format(scale_ratio))
    return scale_ratio, delta_w, delta_w_actual, w_retain


def trapezium_trick(model, model_init, delta_w, w_retain):
    m_pred_error = vectorize_params(model)-vectorize_params(model_init)-w_retain.squeeze()
    print(f"Delta w -------: {np.linalg.norm(delta_w)}")

    inner = np.inner(delta_w/np.linalg.norm(delta_w),m_pred_error/np.linalg.norm(m_pred_error))
    print(f"Inner Product--: {inner}")

    if inner<0:
        angle = np.arccos(inner)-np.pi/2
        print(f"Angle----------:  {angle}")

        predicted_norm=np.linalg.norm(delta_w) + 2*np.sin(angle)*np.linalg.norm(m_pred_error)
        print(f"Pred Act Norm--:  {predicted_norm}")
    else:
        angle = np.arccos(inner) 
        print(f"Angle----------:  {angle}")

        predicted_norm=np.linalg.norm(delta_w) + 2*np.cos(angle)*np.linalg.norm(m_pred_error)
        print(f"Pred Act Norm--:  {predicted_norm}")

    predicted_scale=predicted_norm/np.linalg.norm(delta_w)
    print(f"Predicted Scale:  {predicted_scale}")
    return predicted_scale

# Normalized Inner Product between Prediction and Actual Scrubbing Update
def NIP(v1,v2):
    nip = (np.inner(v1/np.linalg.norm(v1),v2/np.linalg.norm(v2)))
    print(nip)
    return nip


def get_delta_w_dict(delta_w,model):
    # Give normalized delta_w
    delta_w_dict = OrderedDict()
    params_visited = 0
    for k,p in model.named_parameters():
        num_params = np.prod(list(p.shape))
        update_params = delta_w[params_visited:params_visited+num_params]
        delta_w_dict[k] = torch.Tensor(update_params).view_as(p)
        params_visited+=num_params
    return delta_w_dict

def get_ntk_model(device, model, predicted_scale, delta_w):
    scale=predicted_scale
    direction = get_delta_w_dict(delta_w,model)

    model_scrub = copy.deepcopy(model)
    for k,p in model_scrub.named_parameters():
        p.data += (direction[k]*scale).to(device)
    return model_scrub

# combination of deepcopy and hessian() in Golatkar's notebook
def fisher_init(device, dataset, model):
    modelf = copy.deepcopy(model)
    for p in modelf.parameters():
        p.data0 = copy.deepcopy(p.data.clone())

    modelf.eval()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()

    for p in modelf.parameters():
        p.grad_acc = 0
        p.grad2_acc = 0
    
    for data, orig_target in tqdm(train_loader):
        data, orig_target = data.to(device), orig_target.to(device)
        output = modelf(data)
        prob = F.softmax(output, dim=-1).data

        for y in range(output.shape[1]):
            target = torch.empty_like(orig_target).fill_(y)
            loss = loss_fn(output, target)
            modelf.zero_grad()
            loss.backward(retain_graph=True)
            for p in modelf.parameters():
                if p.requires_grad:
                    p.grad_acc += (orig_target == target).float() * p.grad.data
                    p.grad2_acc += prob[:, y] * p.grad.data.pow(2)
    for p in modelf.parameters():
        p.grad_acc /= len(train_loader)
        p.grad2_acc /= len(train_loader)
    return modelf

def get_mean_var(p, num_classes, num_to_forget, class_to_forget, is_base_dist=False, alpha=3e-6):
    var = copy.deepcopy(1./(p.grad2_acc+1e-8))
    var = var.clamp(max=1e3)
    if p.size(0) == num_classes:
        var = var.clamp(max=1e2)
    var = alpha * var
    
    if p.ndim > 1:
        var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
    if not is_base_dist:
        mu = copy.deepcopy(p.data0.clone())
    else:
        mu = copy.deepcopy(p.data0.clone())
    if p.size(0) == num_classes and num_to_forget is None:
        mu[class_to_forget] = 0
        var[class_to_forget] = 0.0001
    if p.size(0) == num_classes:
        # Last layer
        var *= 10
    elif p.ndim == 1:
        # BatchNorm
        var *= 10
#         var*=1
    return mu, var

def kl_divergence_fisher(mu0, var0, mu1, var1):
    return ((mu1 - mu0).pow(2)/var0 + var1/var0 - torch.log(var1/var0) - 1).sum()

def get_info_left(seed, num_classes, num_to_forget, modelf, modelf0):
    # Computes the amount of information not forgotten at all layers using the given alpha
    alpha = 1e-6
    total_kl = 0
    torch.manual_seed(seed)
    for (k, p), (k0, p0) in zip(modelf.named_parameters(), modelf0.named_parameters()):
        mu0, var0 = get_mean_var(p, num_classes, num_to_forget, False, alpha=alpha)
        mu1, var1 = get_mean_var(p0, num_classes, num_to_forget, True, alpha=alpha)
        kl = kl_divergence_fisher(mu0, var0, mu1, var1).item()
        total_kl += kl
        print(k, f'{kl:.1f}')
    print("Total:", total_kl)
    return total_kl

def apply_fisher_noise(seed, num_classes, num_to_forget, modelf, modelf0):
    #fisher_dir = [] - Unused in golatkar's notebook
    alpha = 1e-6
    torch.manual_seed(seed)
    for i, p in enumerate(modelf.parameters()):
        mu, var = get_mean_var(p, num_classes, num_to_forget, False, alpha=alpha)
        p.data = mu + var.sqrt() * torch.empty_like(p.data0).normal_()
        #fisher_dir.append(var.sqrt().view(-1).cpu().detach().numpy())

    for i, p in enumerate(modelf0.parameters()):
        mu, var = get_mean_var(p, num_classes, num_to_forget, False, alpha=alpha)
        p.data = mu + var.sqrt() * torch.empty_like(p.data0).normal_()

def l2_penalty(model,model_init,weight_decay):
    l2_loss = 0
    for (k,p),(k_init,p_init) in zip(model.named_parameters(),model_init.named_parameters()):
        if p.requires_grad:
            l2_loss += (p-p_init).pow(2).sum()
    l2_loss *= (weight_decay/2.)
    return l2_loss

def run_train_epoch(model: nn.Module, model_init, data_loader: torch.utils.data.DataLoader, 
                    loss_fn: nn.Module,
                    optimizer: torch.optim.SGD, split: str, epoch: int, wt_decay, ignore_index=None,
                    negative_gradient=False, negative_multiplier=-1, random_labels=False,
                    quiet=False,delta_w=None,scrub_act=False):
    model.eval()
    metrics = AverageMeter()    
    num_labels = data_loader.dataset.targets.max().item() + 1
    
    with torch.set_grad_enabled(split != 'test'):
        for idx, batch in enumerate(tqdm(data_loader, leave=False)):
            batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
            input, target = batch
            output = model(input)
            # Never used block of code in Golatkar's notebook
            '''
            if split=='test' and scrub_act:
                G = []
                for cls in range(num_classes):
                    grads = torch.autograd.grad(output[0,cls],model.parameters(),retain_graph=True)
                    grads = torch.cat([g.view(-1) for g in grads])
                    G.append(grads)
                grads = torch.autograd.grad(output_sf[0,cls],model_scrubf.parameters(),retain_graph=False)
                G = torch.stack(G).pow(2)
                delta_f = torch.matmul(G,delta_w)
                output += delta_f.sqrt()*torch.empty_like(delta_f).normal_()
            '''
            loss = loss_fn(output, target) + l2_penalty(model,model_init, wt_decay)
            metrics.update(n=input.size(0), loss=loss_fn(output,target).item(), error=get_error(output, target))
            
            if split != 'test':
                model.zero_grad()
                loss.backward()
                optimizer.step()
    if not quiet:
        log_metrics(split, metrics, epoch)
    return metrics.avg

def finetune_cat_forget(model: nn.Module, data_loader: torch.utils.data.DataLoader, wt_decay, lr=0.01, epochs=10, quiet=False):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0)
    model_init=copy.deepcopy(model)
    for epoch in range(epochs):
        run_train_epoch(model, model_init, data_loader, loss_fn, optimizer, 'train', epoch, wt_decay, ignore_index=None, quiet=quiet)

def test(epoch, model, data_loader, wt_decay):
    loss_fn = nn.CrossEntropyLoss()
    model_init=copy.deepcopy(model)
    return run_train_epoch(model, model_init, data_loader, loss_fn, optimizer=None, split='test', epoch=epoch, wt_decay=wt_decay, ignore_index=None, quiet=True)

def readout_retrain(model, data_loader, test_loader, seed, wt_decay, lr=0.01, epochs=100, threshold=0.01, quiet=True):
    torch.manual_seed(seed)
    model = copy.deepcopy(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0)
    sampler = torch.utils.data.RandomSampler(data_loader.dataset, replacement=True, num_samples=500)
    data_loader_small = torch.utils.data.DataLoader(data_loader.dataset, batch_size=data_loader.batch_size, sampler=sampler, num_workers=data_loader.num_workers)
    metrics = []
    model_init=copy.deepcopy(model)
    for epoch in range(epochs):
        metrics.append(run_train_epoch(model, model_init, test_loader, loss_fn, optimizer, 
                                        split='test', epoch=epoch, wt_decay=wt_decay, ignore_index=None, quiet=quiet))
        if metrics[-1]['loss'] <= threshold:
            break
        run_train_epoch(model, model_init, data_loader_small, loss_fn, optimizer, split='train', epoch=epoch,
                         wt_decay=wt_decay, ignore_index=None, quiet=quiet)
    return epochs - 1, metrics

def all_readouts(epoch, seed, model, train_loader, forget_loader, retain_loader, test_loader_full, 
                train_loader_full, batch_size, wt_decay, thresh=0.1, name='method'):

    train_loader = torch.utils.data.DataLoader(train_loader_full.dataset, batch_size, shuffle=True)
    retrain_time, _ = readout_retrain(model, train_loader, forget_loader, seed, wt_decay = wt_decay, epochs=100, lr=0.01, threshold=thresh)
    test_error = test(epoch, model, test_loader_full, wt_decay)['error']
    forget_error = test(epoch, model, forget_loader, wt_decay)['error']
    retain_error = test(epoch, model, retain_loader, wt_decay)['error']
    print(f"{name} ->"
          f"\tFull test error: {test_error:.2%}"
          f"\tForget error: {forget_error:.2%}\tRetain error: {retain_error:.2%}"
          f"\tFine-tune time: {retrain_time+1} steps")
    return dict(test_error=test_error, forget_error=forget_error, retain_error=retain_error, retrain_time=retrain_time)

def test_on_loaders(epoch, model, loaders, wt_decay):
    for loader in loaders:
        test(epoch, model, loader, wt_decay)

def extract_retrain_time(metrics, threshold=0.1):
    losses = np.array([m['loss'] for m in metrics])
    return np.argmax(losses < threshold)