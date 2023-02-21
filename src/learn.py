#!/usr/bin/env python3

# pyright: reportMissingImports=true, reportUntypedBaseClass=true, reportGeneralTypeIssues=true
import argparse
import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import models
import load_datasets
import pickle
from utils import AverageMeter, get_logger, get_error, seed_everything, set_batchnorm_mode,\
                    cutmix_data, mkdir, save_model, trainable_params_, load_pretrained

# def adjust_learning_rate(args, optimizer, epoch):
#     if args.step_size is not None:lr = args.lr * 0.1 ** (epoch//args.step_size)
#     else:lr = args.lr
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
        
# def l2_penalty(model,model_init,weight_decay):
#     l2_loss = 0
#     for (k,p),(k_init,p_init) in zip(model.named_parameters(),model_init.named_parameters()):
#         if p.requires_grad:
#             l2_loss +=  (p-p_init).pow(2).sum()
#     l2_loss *= (weight_decay/2.)
#     return l2_loss
    
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None)
    parser.add_argument('--log-dir', default='.', 
                        help='Folder where all logs are stored')
    parser.add_argument('--exp-name', default='LastExpt', 
                        help='Subfolder where all logs are stored') 
    parser.add_argument('--procedure', default='train', 
                        help='Suffix to identify checkpoint')
    parser.add_argument('--confname', default=None,
                        help='Confusion description for checkpoint')
    parser.add_argument('--logname', default='train',
                        help='name of output log file')

    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--model', default='resnet20')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of Classes')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint to resume')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--validsplit', type=int, default=0.2, metavar='S',
                        help='train-valid split ratio (default: 0.2)')

    parser.add_argument('--epochs', type=int, default=62, metavar='N',
                        help='number of epochs to train (default: 62)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--maxlr', type=float, default=0.1, metavar='LR',
                        help='max learning rate for SGDR (default: 0.1)')
    parser.add_argument('--minlr', type=float, default=0.005, metavar='LR',
                        help='min learning rate for SGDR (default: 0.005)')
    
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-5, metavar='M',
                        help='Weight decay (default: 5e-5)')
    parser.add_argument('--regularization', default=None,
                        help='Regularization type (default: None)')
    parser.add_argument('--scheduler', default='CosineAnnealingWarmRestarts',
                        choices = ['CosineAnnealingWarmRestarts', 'CosineAnnealingLR'],
                        help='Pytorch Scheduler name: (default: CosineAnnealingWarmRestarts')
    parser.add_argument('--cutmix-prob', type=float, default=0.5, metavar='P',
                        help='Prob. for cutmix (default: 0.5)')
    parser.add_argument('--cutmix-alpha', type=float, default=1.0, metavar='A',
                        help='Alpha for cutmix (default: 1.0)')
    parser.add_argument('--clip', type=float, default=10.0, metavar='M',
                        help='Gradient clipping (default: 10)')

    parser.add_argument('--forget-class', type=int, default=None,
                        help='Class to forget')
    parser.add_argument('--num-to-forget', type=int, default=0,
                        help='Number of samples of class to forget')
    parser.add_argument('--confusion', default=None,
                        help='Confusion matrix for mislabelling')
    parser.add_argument('--confusion-copy', default=False,
                        help='Whether confusion is via copying')
    parser.add_argument('--forget-indices', default=None,
                        help='Indices to be forgotten')

    parser.add_argument('--disable-bn', action='store_true', default=False,
                        help='Put batchnorm in eval mode and don\'t update the running averages')
    parser.add_argument('--lossfn', type=str, default='ce',
                        help='Cross Entropy: ce or mse')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--step-size', default=None, type=int, help='learning rate scheduler')
    args = parser.parse_args()
    return args

def argsName(args):
    if args.name is None:
        args.name = f"{args.dataset}_{args.model}_seed-{args.seed}"
        if args.num_to_forget is not None:
            args.name += f"_Nf-{args.num_to_forget}"
        if args.forget_class is not None:
            args.name += f"_Cf-{args.forget_class}"
        if args.confusion is not None:
            args.confname += f"_conf-{args.confname}"
        args.name += f"_split-{str(args.validsplit)}"

        args.name+=f"_ep-{args.epochs}"
        args.name+=f"_bs-{args.batch_size}"
        args.name+=f"_lr-[{str(args.minlr).replace('.','_')}-{str(args.maxlr).replace('.','_')}]"
        args.name+=f"_wd-{str(args.weight_decay).replace('.','_')}"
        if args.regularization is not None:
            args.name+=f"_{args.regularization}"
        if args.regularization=="cutmix":
            args.name+=f"_cmixp-{str(args.cutmix_prob).replace('.','_')}"
            args.name+=f"_cmixa-{str(args.cutmix_alpha).replace('.','_')}"
        if args.scheduler!="CosineAnnealingWarmRestarts":
            args.name+=f"_sched-{args.scheduler}"
    return args

def argsNf(args):
    if args.confusion is not None:
        args.num_to_forget = 0
        for i in range(len(args.confusion)):
            for j in range(len(args.confusion[i])):
                if i != j:  args.num_to_forget += args.confusion[i][j]
    return args

def train(args, model, loader, criterion, optimizer, epoch, logger):
    model.train()
    losses, data_time, batch_time = AverageMeter(), AverageMeter(), AverageMeter()
    start = time.time()

    if args.disable_bn:
        set_batchnorm_mode(model, train=False)

    for inputs, labels in loader:
        # Tweak inputs
        inputs, labels = inputs.cuda(non_blocking=True), (labels).cuda(non_blocking=True)
        #if args.dataset == 'MNIST': inputs=inputs.view(inputs.shape[0]*inputs.shape[1],-1)

        do_cutmix = args.regularization == 'cutmix' and np.random.rand(1) < args.cutmix_prob
        cutmix_lam, cutmix_labels_a, cutmix_labels_b = None, None, None
        if do_cutmix:
            inputs, cutmix_labels_a, cutmix_labels_b, cutmix_lam = cutmix_data(x=inputs, y=labels, alpha=args.cutmix_alpha)
        data_time.update(time.time() - start)
        
        # Forward, backward passes then step
        outputs = model(inputs)
        loss = None
        if do_cutmix:
            loss = cutmix_lam * criterion(outputs, cutmix_labels_a) + (1 - cutmix_lam) * criterion(outputs, cutmix_labels_b)
        else:
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip) # Always be safe than sorry
        optimizer.step()

        # Log losses
        losses.update(loss.data.item(), labels.size(0))
        batch_time.update(time.time() - start)
        start = time.time()
    logger.info('==> Train:[{0}]\tTime:{batch_time.sum:.4f}\tData:{data_time.sum:.4f}\tLoss:{loss.avg:.4f}\t'.format(epoch, batch_time=batch_time, data_time=data_time, loss=losses))
    return model, optimizer, losses.avg

def test(args, model, loader, mode, criterion, epoch, logger):
    model.eval()
    losses, batch_time, errors = AverageMeter(), AverageMeter(), AverageMeter()

    with torch.no_grad():
        start = time.time()
        for inputs, labels in loader:
            # Get outputs
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.update(loss.data, inputs.size(0))
            
            errors.update(get_error(outputs, labels))

            batch_time.update(time.time() - start)
            start = time.time()

        logger.info('==> {1}: [{0}]\tTime:{batch_time.sum:.4f}\tLoss:{losses.avg:.4f}\tError:{errors.avg:.4f}\t'
            .format(epoch, mode, batch_time=batch_time, losses=losses, errors=errors))
    return errors.avg

def train_loop(model, args, epochs, logger, train_loader, valid_loader, test_loader, prefixpath):
    MAX_LOSS = 1E8
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(trainable_params_(model), lr=args.maxlr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=args.minlr)
    elif args.scheduler == 'CosineAnnealingLR':
        #[TODO]: Set T_max
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=args.minlr)
    else:
        raise SystemExit('Scheduler not found')

    best_err, best_loss, best_model = 1.0, MAX_LOSS, copy.deepcopy(model)
    #global epoch
    train_time = 0
    for epoch in range(epochs):
        logger.info("==> Starting pass number: "+str(epoch)+", Learning rate: " + str(optimizer.param_groups[0]['lr']))
        start_time = time.monotonic()
        model, optimizer, tr_loss = train(args, model, train_loader, criterion, optimizer, epoch, logger)
        train_time  += time.monotonic() - start_time
        err = test(args, model, valid_loader, 'Valid', criterion, epoch, logger)
        test(args, model, test_loader, 'Test', criterion, epoch, logger)
        best_err, best_model = err, model #Just take last model
        scheduler.step()
        
        #epoch+2 should be a multiple of 2, after that warm restarts
        if ((epoch+2) & (epoch+1) == 0) and epoch >= 6:
            save_model(f'{prefixpath}_{epoch}eps.pt', model)

        ''' #Uncomment if need best training loss model
        if tr_loss < best_loss:
            logger.info('==> Best Train Loss\tPrevious: [{:.3f}]\t'.format(best_loss) + 'Current: [{:.3f}]\t'.format(tr_loss))
            best_loss = tr_loss
            best_err = err
            best_model = copy.deepcopy(model)
        '''
    
    logger.info('==> Training completed! Valid Error: [{0:.3f}]'.format(best_err))
    # save_model(f'{args.log_dir}/{args.exp_name}/{args.name}.pt', best_model)

    return best_model, train_time


def learn(args):
    mkdir(args.log_dir+'/'+args.exp_name) 
    args = argsName(argsNf(args)) #Get args.num_to_forget if confusion, args.name
    console_logger = get_logger(folder=args.log_dir+'/'+args.exp_name+'/', logname=args.logname)
    console_logger.info(f'Checkpoint name: {args.name}_{args.procedure}')
    
    seed_everything(args.seed)
    #global train_loader, valid_loader, test_loader
    console_logger.debug(f'C_f: {args.forget_class}\t N_f: {args.num_to_forget}\t indexes_f: {args.forget_indices}\t'
                        f' confusion: {args.confusion}')
    train_loader, valid_loader, test_loader, confu_idxs = load_datasets.get_loaders(console_logger, args.dataset, class_to_replace=args.forget_class,
                                                     validsplit=args.validsplit, num_indexes_to_replace=args.num_to_forget, 
                                                     indexes_to_replace=args.forget_indices, batch_size=args.batch_size, 
                                                     seed=args.seed, confusion=args.confusion, confusion_copy=args.confusion_copy)
    
    num_classes = max(train_loader.dataset.targets) + 1 if args.num_classes is None else args.num_classes
    args.num_classes = num_classes
    console_logger.debug(f"Number of Classes: {num_classes}")
    console_logger.debug(f"Train set size: {len(train_loader.dataset.targets)} | "
                        f"Valid set size: {len(valid_loader.dataset.targets)} | Test set size: {len(test_loader.dataset.targets)}")
    model = models.get_model(args.model, num_classes=num_classes).to('cuda')
    
    #TODO: Do we really need to load pretrained in this weird way? I doubt it...
    if args.resume is not None:
        console_logger.info(f"Loading pretrained model at {args.resume}")
        model = load_pretrained(model, args.model, args.resume)
    
    #TODO: Pretrained model save  in state_dict() format needed for golatkar's methods, change to our load_ save_ calls everywhere
    model_init = copy.deepcopy(model)
    prefixpath = f"{args.log_dir}/{args.exp_name}/{args.name}"
    torch.save(model_init.state_dict(), f"{prefixpath}_init.pt")
    best_model, train_time = train_loop(model, args, args.epochs, console_logger, 
                            train_loader, valid_loader, test_loader, f"{prefixpath}_{args.procedure}")
    return console_logger, train_loader, valid_loader, test_loader, confu_idxs, best_model, args.epochs - 1, train_time

'''Shash: added new only-callable-not-runnable arguments for confusion test'''
def caller(args_dict):
    argumentlist = ['batch_size', 'dataset', 'disable_bn', 'epochs', 'seed', 
    'logname', 'lossfn', 'minlr', 'maxlr', 'model', 'momentum', 'no_cuda',
    'num_classes', 'log_dir', 'name', 'exp_name', 'resume', 'procedure', 
    'step_size', 'weight_decay', 'validsplit', 'scheduler',
    'regularization', 'cutmix_prob', 'cutmix_alpha', 'clip',
    'forget_class', 'confname', 'confusion', 'confusion_copy', 'forget_indices', 'num_to_forget']
    global args
    args = argparse.Namespace(
        batch_size = int(64),
        dataset = 'cifar10',
        disable_bn = False,
        epochs = int(62),
        forget_class = None,
        lossfn = 'ce',
        minlr = 0.005,
        maxlr = 0.1, 
        model = 'resnet20',
        momentum = 0.9,
        no_cuda = False,
        num_classes = None,
        num_to_forget = 0,
        log_dir = '.',
        name = None,
        exp_name = 'LastExpt',
        resume = None,
        procedure = 'original',
        seed = int(1),
        step_size = None, #Only old golatkar stuff
        weight_decay = 5e-5,
        regularization = None,
        scheduler = 'CosineAnnealingWarmRestarts',
        cutmix_prob = 0.5,
        cutmix_alpha = 1.0,
        clip = 10.0,
        confname = None,
        confusion = None,
        confusion_copy = False,
        forget_indices = None,
        logname = 'train',
        validsplit = 0.2,
    )
    dictargs = vars(args)
    for arg in argumentlist:
        if arg in args_dict.keys():
            #print(arg)
            dictargs[arg] = args_dict[arg]
    logger, train_loader, valid_loader, test_loader, confu_idxs, model, epoch, train_time = learn(args)
    return args, logger, train_loader, valid_loader, test_loader, confu_idxs, model, epoch, train_time

if __name__ == '__main__':
    args = parse_args()
    learn(args)