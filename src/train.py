#!/usr/bin/env python3

# pyright: reportMissingImports=true, reportUntypedBaseClass=true, reportGeneralTypeIssues=true

import os, time, argparse, pickle, random
from learn import caller
from utils import get_logger, mkdir, save_model

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--logname', default='train',
                        help='name of output log file')
    parser.add_argument('--log-dir', default='.', 
                        help='Folder where all logs are stored (default: .)')
    parser.add_argument('--exp-name', default='LastExpt', 
                        help='Subfolder where all logs are stored (default: LastExpt)') 

    parser.add_argument('--retrain-scratch', default=True,
                        help='Whether to try retrain from scratch (default: True)')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model', default='resnet20')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='Number of Classes')
    parser.add_argument('--validsplit', type=int, default=0.2, metavar='S',
                        help='train-valid split ratio (default: 0.2)')

    parser.add_argument('--path-pre', default=None, metavar='P',
                        help='Pretrain checkpoint (default: None)')
    parser.add_argument('--epochs-og', type=int, default=62, metavar='N',
                        help='number of epochs to train (default: 62)')
    parser.add_argument('--batch-og', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--maxlr-og', type=float, default=0.1, metavar='LR',
                        help='max learning rate for SGDR (default: 0.1)')
    parser.add_argument('--minlr-og', type=float, default=0.005, metavar='LR',
                        help='min learning rate for SGDR (default: 0.005')
    parser.add_argument('--regularization', default=None,
                        choices = ['none', 'cutmix'],
                        help='Regularization type (default: None)')
    parser.add_argument('--scheduler', default='CosineAnnealingWarmRestarts',
                        choices = ['CosineAnnealingWarmRestarts', 'CosineAnnealingLR'],
                        help='Pytorch Scheduler name: (default: CosineAnnealingWarmRestarts')
    parser.add_argument('--forget-class', type=int, default=None,
                        help='Class to forget (default: None)')
    parser.add_argument('--num-to-forget', type=int, default=0,
                        help='Number of samples of class to forget (default: 0)')

    parser.add_argument('--confname', default=None,
                        help='Confusion (None if no confusion) description for checkpoint (default: None)')
    parser.add_argument('--conftype', default='exchange',
                        help='Confusion type (default: exchange)')
    parser.add_argument('--num-change', type=int, default=10,
                        help='No. of samples per class to exch (default: 10)')
    parser.add_argument("--exch-classes", nargs="+", default=None, 
                        type=int, help='List of classes to exchange space separated')

    args = parser.parse_args()
    if args.regularization == 'none':
        args.regularization = None
    return args


def finetune(mod_name, dataset, exp_name, log_dir, resumePath, logname='output', validsplit=0.2,
            maxlr=0.01, minlr=0.0005, epochs=62, batch=64, seed=1,
            scheduler=None, regularization=None, confusion = None, confusion_copy=False, confname=None):
    args_dict = {"dataset":dataset, "model":mod_name, "maxlr":maxlr, "minlr":minlr, "logname":logname,
             "resume":resumePath, "log_dir":log_dir, "exp_name":exp_name, "procedure":"original",
             "disable_bn":False, "epochs":epochs, "seed":seed, "batch_size":batch, "weight_decay":5e-5, #ideally: 5e-5, golat 0.1
             "regularization": regularization,"scheduler":scheduler,
             "confname":confname, "confusion":confusion,"confusion_copy":confusion_copy, "validsplit":validsplit}
    args, logger, _, _, _, confu_idxs, model, epoch, train_time = caller(args_dict)
    return args, logger, confu_idxs, model, epoch, train_time

#ENSURE SAME SEED AS FINETUNE
def retrain_from_scratch(mod_name, dataset, exp_name, log_dir, resumePath, logname='output',
                        maxlr=0.01, minlr=0.0005, epochs=62, batch=64, seed=1,
                        scheduler=None, regularization=None, C_f=None, N_f=0, confusion_indices=None):
    args_dict = {"dataset":dataset, "model":mod_name, "maxlr":maxlr, "minlr":minlr, "logname":logname,
                "weight_decay":5e-5, "disable_bn":False, "epochs":epochs, "seed":seed, "batch_size":batch, #ideally: decay 5e-5
                "regularization": regularization, "scheduler":scheduler,
             "resume":resumePath, "log_dir":log_dir, "exp_name":exp_name, "procedure":"retrain",
             "forget_class":C_f, "num_to_forget":N_f, "forget_indices":confusion_indices}
    args, logger, _, _, _, confu_idxs, model, epoch, train_time = caller(args_dict)
    return args, logger, model, epoch, train_time

#Modify based on need for confusion
def confmat(n_C, conftype, num_change, exch_classes=None):
    mat = []
    for i in range(n_C):
        mat.append([])
        for j in range(n_C):
            mat[i].append(0)

    if conftype == 'noise':
        for i in range(n_C):
            for j in range(n_C):
                if i != j:  mat[i][j] = int(num_change/(n_C-1))
            leftover = num_change % n_C
            c_t = [j for j in range(n_C) if j!=i]
            for j in random.sample(c_t, leftover):
                mat[i][j] += 1

    if conftype == 'exchange':
        assert(exch_classes is not None)
        for i in exch_classes:
            for j in exch_classes:
                if i != j:  mat[i][j] = num_change

    return mat

if __name__ == "__main__":
    args = parse_args()
    #########################################################################################################
    #Beyond this point only edit call parameters to finetune and retrain_from_scratch

    mkdir(f"{args.log_dir}/{args.exp_name}")

    logger = get_logger(folder=args.log_dir+'/'+args.exp_name+'/', logname=args.logname)

    confusion = None
    if args.confname is not None:
        confusion = confmat(args.num_classes, args.conftype, args.num_change, args.exch_classes)
    else:
        args.exch_classes = None

    args_og, logger, confu_idxs, model_og, epoch, compute_og\
        = finetune(args.model, args.dataset, args.exp_name, args.log_dir, args.path_pre, maxlr=args.maxlr_og, minlr=args.minlr_og, 
                    validsplit=args.validsplit, batch=args.batch_og, logname=args.logname, epochs=args.epochs_og, 
                    scheduler=args.scheduler, regularization=args.regularization, 
                    confusion=confusion, confname=args.confname, seed=args.seed)

    path_o = f'{args_og.log_dir}/{args_og.exp_name}/{args_og.name}_{args_og.procedure}.pt'
    save_model(path_o, model_og)
    if args.confname is not None:  
        C_f = None
        N_f = args_og.num_to_forget

    path_oarg = f"{args.log_dir}/{args.exp_name}/args_og.txt"
    pickle.dump(args_og, open(path_oarg, 'wb'))

    if not args.retrain_scratch:
        exit(0)

    if args.confname is None:
        C_f = args.forget_class
        N_f = args.num_to_forget
        
    args_re, logger, model_re, epoch, compute_re =\
        retrain_from_scratch(args.model, args.dataset, args.exp_name, args.log_dir, args.path_pre, logname=args.logname,
                             batch=args.batch_og, maxlr = args.maxlr_og, minlr=args.minlr_og, epochs=args.epochs_og, 
                             scheduler=args.scheduler, regularization=args.regularization, 
                             confusion_indices=confu_idxs, C_f=C_f, N_f=N_f, seed=args.seed)

    path_r = f'{args_re.log_dir}/{args_re.exp_name}/{args_re.name}_{args_re.procedure}.pt'
    save_model(path_r, model_re)
    init_checkpoint = f"{args_re.log_dir}/{args_re.exp_name}/{args_re.name}_init.pt"

    path_rarg = f"{args.log_dir}/{args.exp_name}/args_re.txt"
    pickle.dump(args_re, open(path_rarg, 'wb'))

    logger.info(f'Retain/Forget extraction\n C_f: {args_re.forget_class}\n N_f: {args_re.num_to_forget}\n'
                        f' seed: {args_re.seed}\n confu_idxs: {confu_idxs}\n batch: {args_re.batch_size}')
    logger.info(f'Times - Original: {compute_og} \nRetrain: {compute_re}')
    with open(f'{args.log_dir}/{args.exp_name}/train-paths.txt', 'w') as f:
        print(f"{path_oarg}\n{path_rarg}\n{path_o}\n{path_r}\n{init_checkpoint}", file=f)