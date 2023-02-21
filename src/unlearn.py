#!/usr/bin/env python3

# pyright: reportMissingImports=true, reportUntypedBaseClass=true, reportGeneralTypeIssues=true

import os, time, argparse, copy, pickle, torch
from load_datasets import get_loaders
from golatkar import golatkar_precomputation, get_ntk_model, fisher_init, \
                    apply_fisher_noise
from methods import retrain_lastK, cat_forget_finetune
from utils import argparse2bool, load_model, print_model_params, save_model, get_forget_retain_loader, print_final_times, get_logger, mkdir
from models import get_model

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--oldLoad', type=argparse2bool, nargs='?',
                        const=True, default=False,
                        help='Old (golatkar) styled checkpoints (default: False)')
    parser.add_argument('--path-o', required=True,
                        help='Path to original model checkpoint')
    parser.add_argument('--path-r', required=True,
                        help='Path to retrained model checkpoint')
    parser.add_argument('--path-oarg', required=True,
                        help='Path to original model args')
    parser.add_argument('--path-rarg', required=True,
                        help='Path to retrained model args')
    parser.add_argument('--init-checkpoint', required=True,
                        help='Path to init checkpoint for golatkar')

    parser.add_argument('--num-classes', type=int, default=10,
                        help='Number of Classes')
    parser.add_argument('--scheduler', default=None,
                        choices = ['CosineAnnealingWarmRestarts', 'CosineAnnealingLR'],
                        help='Pytorch Scheduler name: (default: The one used for train, in args_re')
    parser.add_argument('--regularization', default=None,
                        choices = ['none', 'cutmix', 'remove'],
                        help='Regularization type (default: None)')

    parser.add_argument('--golatkar', type=argparse2bool, nargs='?',
                        const=True, default=False,
                        help='Whether to use golatkar methods')
    parser.add_argument('--name-go', default='Golatkar')
    
    parser.add_argument('--retrfinal', type=argparse2bool, nargs='?',
                        const=True, default=False,
                        help='Whether to use retrain final K')
    parser.add_argument('--name-rf', default='RetrFinal')
    parser.add_argument('--epochs-rf', type=int, default=62, metavar='N',
                        help='number of epochs to train (default: 62)')
    parser.add_argument('--maxL-rf', type=int, default=3, metavar='UL',
                        help='Layers to retrain upperbound (default: 3)')
    parser.add_argument('--minL-rf', type=int, default=1, metavar='LL',
                        help='Layers to retrain lowerbound (default: 1)')
    parser.add_argument('--stepL-rf', type=int, default=1, metavar='LS',
                        help='Layers to retrain step size (default: 1)')

    parser.add_argument('--finetune', type=argparse2bool, nargs='?',
                        const=True, default=False,
                        help='Whether to use finetune method')    
    parser.add_argument('--name-ft', default='Finetune')
    parser.add_argument('--epochs-ft', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 62)')
    parser.add_argument('--maxlr-ft', type=float, default=0.1, metavar='LR',
                        help='max learning rate for SGDR (default: 0.1)')
    parser.add_argument('--minlr-ft', type=float, default=0.005, metavar='LR',
                        help='min learning rate for SGDR (default: 0.005')

    parser.add_argument('--finetune-final', type=argparse2bool, nargs='?',
                        const=True, default=False,
                        help='Whether to use finetune final K')
    parser.add_argument('--name-ftF', default='FinetuneFinal')
    parser.add_argument('--epochs-ftF', type=int, default=62, metavar='N',
                        help='number of epochs to train (default: 62)')
    parser.add_argument('--maxL-ftF', type=int, default=3, metavar='UL',
                        help='Layers to finetune upperbound (default: 3)')
    parser.add_argument('--minL-ftF', type=int, default=1, metavar='LL',
                        help='Layers to finetune lowerbound (default: 1)')
    parser.add_argument('--stepL-ftF', type=int, default=1, metavar='LS',
                        help='Layers to finetune step size (default: 1)')

    args = parser.parse_args()
    if args.regularization == 'none':
        args.regularization = None
    return args

def update_args(args):
    d = vars(args)

    if 'scheduler' not in d.keys():
        d['scheduler'] = 'CosineAnnealingWarmRestarts'

    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    args_og = pickle.load(open(args.path_oarg, 'rb'))
    args_re = pickle.load(open(args.path_rarg, 'rb'))
    args_og = update_args(args_og)
    args_re = update_args(args_re)

    scheduler_unlearn = args_re.scheduler
    if args.scheduler is not None: #override scheduler for unlearning
        scheduler_unlearn = args.scheduler
    regularization_unlearn = args_re.regularization
    if args.regularization is not None: #if None, take original.
        if args.regularization == 'remove': #remove training regularizer
            regularization_unlearn = None 
        else: regularization_unlearn = args.regularization #otherwise switch
    

    log_dir, orig_name, confname, modname = args_og.log_dir, args_og.exp_name, args_og.confname, args_og.model
    pretrain_path = args_og.resume

    logname = 'unlearn'
    mat_folder = 'Matrices'
    prefixpath = f"{log_dir}/{orig_name}"
    mkdir(f"{prefixpath}/{mat_folder}")

    logger_un = get_logger(folder=f'{prefixpath}/', logname=logname)

    logger_un.info(f'Original loader extraction\n C_f: {args_og.forget_class}\n N_f: {args_og.num_to_forget}\n'
                        f'idxrep: {args_og.forget_indices}\n seed: {args_og.seed}\n confu_idxs: {args_og.confusion}\n'
                        f'batch: {args_og.batch_size}\n data: {args_og.dataset}')
    loader_tr, loader_v, loader_te, confu_idxs = \
                get_loaders(logger_un, args_og.dataset, class_to_replace=args_og.forget_class,
                    num_indexes_to_replace=args_og.num_to_forget, indexes_to_replace=args_og.forget_indices,
                    batch_size=args_og.batch_size, seed=args_og.seed,
                    confusion=args_og.confusion, confusion_copy=args_og.confusion_copy, transform_tr = True)

    logger_un.info(f'Retain/Forget extraction\n C_f: {args_re.forget_class}\n N_f: {args_re.num_to_forget}\n'
                        f' seed: {args_re.seed}\n confu_idxs: {confu_idxs}\n batch: {args_re.batch_size}')
    marked_loader, _, _, _ = get_loaders(logger_un, args_re.dataset, class_to_replace=args_re.forget_class, 
                                            num_indexes_to_replace=args_re.num_to_forget, only_mark=True, 
                                            batch_size=1, seed=args_re.seed, shuffle=True, 
                                            indexes_to_replace=confu_idxs, transform_tr = True)
    loader_f, loader_re, _ = get_forget_retain_loader(marked_loader, loader_tr, args_re.seed, args_re.batch_size)
    logger_un.info(f'Forget set labels: {loader_f.dataset.targets}')

    num_total = len(loader_tr.dataset)
    num_to_retain = len(loader_re.dataset)

    name = args_re.name
    seed, device, dataset, lossfn, wt_decay = args_re.seed, 'cuda', args_re.dataset, args_re.lossfn, args_re.weight_decay
    arch, filters, forget_class, num_to_forget = args_re.model, 1., args_re.forget_class, args_re.num_to_forget

    args_rf = argparse.Namespace(device = device, log_dir = log_dir, exp_name = orig_name, name = 'retrfinal',
                                momentum=args_re.momentum, disable_bn = args_re.disable_bn, 
                                regularization = regularization_unlearn, scheduler=scheduler_unlearn,
                                cutmix_prob = args_re.cutmix_prob, cutmix_alpha = args_re.cutmix_alpha,
                                clip = args_re.clip, weight_decay = args_re.weight_decay,
                                maxlr = args_re.maxlr, minlr = args_re.minlr)
    args_ftF = argparse.Namespace(device = device, log_dir = log_dir, exp_name = orig_name, name = 'ft_final',
                                momentum=args_re.momentum, disable_bn = args_re.disable_bn, 
                                regularization = regularization_unlearn, scheduler=scheduler_unlearn,
                                cutmix_prob = args_re.cutmix_prob, cutmix_alpha = args_re.cutmix_alpha,
                                clip = args_re.clip, weight_decay = args_re.weight_decay,
                                maxlr = args_re.maxlr, minlr = args_re.minlr)                                
    args_ft = argparse.Namespace(device = device, log_dir = log_dir, exp_name = orig_name, name = 'ft_baseline',
                                momentum=args_re.momentum, disable_bn = args_re.disable_bn,
                                regularization = regularization_unlearn, scheduler=scheduler_unlearn,
                                cutmix_prob = args_re.cutmix_prob, cutmix_alpha = args_re.cutmix_alpha,
                                clip = args_re.clip, weight_decay = args_re.weight_decay,
                                maxlr = args.maxlr_ft, minlr = args.minlr_ft)

    #args used ends here! MODEL EVALUATION BEGINS!
    #################################################################################################################

    model = get_model(args_re.model, num_classes=args.num_classes).to('cuda')
    model_og, model_re = copy.deepcopy(model), copy.deepcopy(model)   
    if args.oldLoad:
        model_og.load_state_dict(torch.load(args.path_o))
        model_re.load_state_dict(torch.load(args.path_r))
    else:
        model_og = load_model(args.path_o, model_og, logger_un)
        model_re = load_model(args.path_r, model_re, logger_un)

    modpath_ntk, modpath_fisher, modpath_ntkfisher, modpath_ft, modprefix_rf, modprefix_ftF =\
         '', '', '', '', '', ''

    # Get jacobian hessian stuff computed and stored
    if args.golatkar:
        mkdir(f'{prefixpath}/{args.name_go}')
        logger_go = get_logger(folder=f'{prefixpath}/{args.name_go}/', logname=logname)
        logger_go.info(f'lossfn: {lossfn}, wt_dec: {wt_decay}\n chkpt: {args.init_checkpoint},'\
            f'#tot: {num_total}, #ret: {num_to_retain}, filters: {filters}')
        starttime = time.monotonic() 
        model_init, scale_ratio, delta_w, delta_w_actual, w_retain, predicted_scale = golatkar_precomputation(
            model_og, model_re, loader_re, loader_f, seed, device, dataset, lossfn, wt_decay,
            args.init_checkpoint, arch, args.num_classes, num_total, num_to_retain, filters, "/scratch/NTK_data")
        compute_prego = time.monotonic() - starttime

        #NTK
        modpath_ntk = f"{prefixpath}/{args.name_go}/ntk.pt"
        modpath_fisher = f"{prefixpath}/{args.name_go}/fisher.pt"
        modpath_ntkfisher = f"{prefixpath}/{args.name_go}/ntkfisher.pt"
        starttime = time.monotonic()
        model_ntk = get_ntk_model(device, model_og, predicted_scale, delta_w)
        compute_ntk = time.monotonic() - starttime
        save_model(modpath_ntk, model_ntk)

        #Fisher 
        starttime = time.monotonic()
        modelf = fisher_init(device, loader_re.dataset, model_og)
        modelf0 = fisher_init(device, loader_re.dataset, model_re)
        apply_fisher_noise(seed, args.num_classes, num_to_forget, modelf, modelf0)
        compute_f = time.monotonic() - starttime
        save_model(modpath_fisher, modelf)

        #NTK+Fisher stuff
        starttime = time.monotonic()
        model_ntkf = fisher_init(device, loader_re.dataset, model_ntk)
        modelf0 = fisher_init(device, loader_re.dataset, model_re)
        apply_fisher_noise(seed, args.num_classes, num_to_forget, model_ntkf, modelf0)
        compute_ntkf = time.monotonic() - starttime
        save_model(modpath_ntkfisher, model_ntkf)
        logger_go.info(f'Times - Precompute: {compute_prego} \nNTK: {compute_ntk} \n'
                        f'Fisher: {compute_f} \nNTK+Fish: {compute_ntkf}')
        logger_go.info(f'NTK Path: {modpath_ntk}')
        logger_go.info(f'Fisher Path: {modpath_fisher}')
        logger_go.info(f'NTKFisher Path: {modpath_ntkfisher}')
        

    #Finetune
    if args.finetune:
        modpath_ft = f"{prefixpath}/{args.name_ft}/{args_ft.name}"
        mkdir(f'{prefixpath}/{args.name_ft}')
        logger_ft = get_logger(folder=f'{prefixpath}/{args.name_ft}/', logname=logname)
        model_ft = copy.deepcopy(model_og)
        model_ft, compute_ft = cat_forget_finetune(args_ft, model_ft, modname, loader_re, loader_v,
                                        loader_te, args.epochs_ft, logger_ft, modpath_ft)
        save_model(f'{modpath_ft}.pt', model_ft)
        logger_ft.info(f'Finetune Time: {compute_ft}')
        logger_ft.info(f'Finetune Path: {modpath_ft}')

    #Retrain last K layers
    if args.retrfinal:
        name_prefix = "RetrFinal_"
        modprefix_rf = f"{prefixpath}/{args.name_rf}/{name_prefix}"
        mkdir(f'{prefixpath}/{args.name_rf}')
        logger_rf = get_logger(folder=f'{log_dir}/{orig_name}/{args.name_rf}/', logname=logname)
        retrfinal_models, retrfinal_times = [], []
        for k in range(args.minL_rf, args.maxL_rf+1, args.stepL_rf):
            model_rf = copy.deepcopy(model_og)
            args_rf.name = f'{name_prefix}{k}'
            model_rf, compute_rf = retrain_lastK(k, args_rf, model_rf, modname, loader_re,
                                    loader_v, loader_te, args.epochs_rf, logger_rf, f'{modprefix_rf}{k}', pretrain_path)
            save_model(f"{modprefix_rf}{k}.pt", model_rf)
            logger_rf.info(f'Retrained last {k} layers. Time: {compute_rf}')
            retrfinal_models.append(model_rf)
        logger_rf.info(f'RetrainFinal Prefix: {modprefix_rf}')

    #Finetune last K layers
    if args.finetune_final:
        name_prefix = "FTfinal_"
        modprefix_ftF = f"{prefixpath}/{args.name_ftF}/{name_prefix}"
        mkdir(f'{prefixpath}/{args.name_ftF}')
        logger_ftF = get_logger(folder=f'{log_dir}/{orig_name}/{args.name_ftF}/', logname=logname)
        FTfinal_models, FTfinal_times = [], []
        for k in range(args.minL_ftF, args.maxL_ftF+1, args.stepL_ftF):
            model_ftF = copy.deepcopy(model_og)
            args_ftF.name = f'{name_prefix}{k}'
            model_ftF, compute_ftF = cat_forget_finetune(args_ftF, model_ftF, modname, loader_re, 
                                    loader_v, loader_te, args.epochs_ftF, logger_ftF, f'{modprefix_ftF}{k}', k)
            save_model(f"{modprefix_ftF}{k}.pt", model_ftF)
            logger_ftF.info(f'Finetuned last {k} layers. Time: {compute_ftF}')
            FTfinal_models.append(model_ftF)
        logger_ftF.info(f'FinetuneFinal Prefix: {modprefix_ftF}')
    
    with open(f'{log_dir}/{orig_name}/unlearn-paths.txt', 'w') as f:
        print(f"{modpath_ntk}\n{modpath_fisher}\n{modpath_ntkfisher}\n{modpath_ft}.pt\n{modprefix_rf}\n{modprefix_ftF}", file=f)