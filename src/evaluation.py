#!/usr/bin/env python3

# pyright: reportMissingImports=true, reportUntypedBaseClass=true, reportGeneralTypeIssues=true
import os, time, argparse, copy, pickle, torch
from load_datasets import get_loaders
from metrics import model_test
from membership import membership_attack
from utils import load_model, print_model_params, get_forget_retain_loader, print_final_times, get_logger, mkdir, argparse2bool, test_loader_split
from models import get_model

#[TODO:] This wrapper is pointless, remove it...
def measure(logger, model, device, matrixpath, loader_re, loader_f, loader_te, 
            met_fname, forget_class, exch_classes, MIA_info=None, loader_te_r=None, loader_te_f=None):
    model_test(logger, model, device, matrixpath, loader_re, loader_f, loader_te,
                met_fname, forget_class, exch_classes, MIA_info, loader_te_r, loader_te_f)
    #output_distance(logger, activations, predictions, ogmod_name, met_sname, f"Retrain_{met_fname}")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--oldLoad', type=argparse2bool, nargs='?',
                        const=True, default=False,
                        help='Old (golatkar) styled checkpoints (default: False)')
    parser.add_argument('--path-o', required=True,
                        help='Path to load original model checkpoint')
    parser.add_argument('--path-r', required=True,
                        help='Path to load retrained model checkpoint')
    parser.add_argument('--path-oarg', required=True,
                        help='Path to load original model args')
    parser.add_argument('--path-rarg', required=True,
                        help='Path to load retrained model args')
    
    parser.add_argument('--MIA-split', type=float, default=0.5,
                        help='Ratio of MIA threshold finding set size')
    parser.add_argument('--MIA-reps', type=int, default=20,
                        help='Number of repetitions to run MIA for stat. signif.')

    parser.add_argument('--no-ogre', type=argparse2bool, nargs='?',
                        const=True, default=False,
                        help='Dont evaluate original and retrain method')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='Number of Classes')
    parser.add_argument("--exch-classes", nargs="+", default=None, 
                        type=int, help='List of classes to exchange space separated')
                        
    parser.add_argument('--custom', type=argparse2bool, nargs='?',
                        const=True, default=False,
                        help='Loading any model by path')
    parser.add_argument('--path-cust', default=None,
                        help='Path to load custom model from')
    parser.add_argument('--result-folder-cust', default='None')    
    parser.add_argument('--logname-cust', default='eval-custom')

    parser.add_argument('--golatkar', type=argparse2bool, nargs='?',
                        const=True, default=False,
                        help='Evaluate golatkar methods')
    parser.add_argument('--name-go', default='Golatkar')
    parser.add_argument('--path-ntk', default=None,
                        help='Path to load NTK model from')
    parser.add_argument('--path-fisher', default=None,
                        help='Path to load Fisher model from')
    parser.add_argument('--path-ntkfisher', default=None,
                        help='Path to load NTK+Fisher model from')
    
    parser.add_argument('--retrfinal', type=argparse2bool, nargs='?',
                        const=True, default=False,
                        help='Evaluate retrain final K')
    parser.add_argument('--name-rf', default='RetrFinal')
    parser.add_argument('--maxL-rf', type=int, default=3, metavar='UL',
                        help='Layers to retrain upperbound (default: 3)')
    parser.add_argument('--minL-rf', type=int, default=1, metavar='LL',
                        help='Layers to retrain lowerbound (default: 1)')
    parser.add_argument('--stepL-rf', type=int, default=1, metavar='LS',
                        help='Layers to retrain step size (default: 1)')
    parser.add_argument('--prefix-rf', default=None,
                        help='Directory to load rf models from')

    parser.add_argument('--finetune-final', type=argparse2bool, nargs='?',
                        const=True, default=False,
                        help='Evaluate finetune final K')
    parser.add_argument('--name-ftF', default='FinetuneFinal')
    parser.add_argument('--maxL-ftF', type=int, default=3, metavar='UL',
                        help='Layers to finetune upperbound (default: 3)')
    parser.add_argument('--minL-ftF', type=int, default=1, metavar='LL',
                        help='Layers to finetune lowerbound (default: 1)')
    parser.add_argument('--stepL-ftF', type=int, default=1, metavar='LS',
                        help='Layers to finetune step size (default: 1)')
    parser.add_argument('--prefix-ftF', default=None,
                        help='Directory to load ftF models from')

    parser.add_argument('--finetune', type=argparse2bool, nargs='?',
                        const=True, default=False,
                        help='Whether to use finetune method')    
    parser.add_argument('--name-ft', default='Finetune')
    parser.add_argument('--path-ft', default=None,
                        help='Path to load finetune model from')


    args = parser.parse_args()
    if args.golatkar:
        if args.path_ntk is None or args.path_fisher is None or args.path_ntkfisher is None:
            parser.error('Golatkar paths (ntk, fisher, ntkfisher) not provided')
    if args.finetune and args.path_ft is None:
        parser.error('Finetune path not provided')
    if args.retrfinal and args.prefix_rf is None:
        parser.error('Retrfinal prefix not provided')
    
    return args

if __name__ == "__main__":
    args=parse_args()
    print(args.path_ft, args.prefix_rf)

    args_og = pickle.load(open(args.path_oarg, 'rb'))
    args_re = pickle.load(open(args.path_rarg, 'rb'))

    log_dir, orig_name, confname, modname = args_og.log_dir, args_og.exp_name, args_og.confname, args_og.model

    logname = 'eval'    
    mat_folder = 'Matrices'
    prefixpath = f"{log_dir}/{orig_name}"
    mkdir(f"{prefixpath}/{mat_folder}")

    if args.no_ogre:
        logger_un = get_logger(folder=f'{prefixpath}/', logname='eval-debug-last')
    else:
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
    loader_f, loader_re, loader_og_f = get_forget_retain_loader(marked_loader, loader_tr, args_re.seed, args_re.batch_size)
    if len(loader_f.dataset.targets) < 500:
        logger_un.info(f'Forget set labels: {loader_f.dataset.targets}')

    num_total = len(loader_tr.dataset)
    num_to_retain = len(loader_re.dataset)

    loader_te_r, loader_te_f = test_loader_split(loader_te, args_re.seed, args_re.batch_size, args.exch_classes, args_re.forget_class)

    noise_expt = False 
    if loader_te_f is None and args_re.forget_class is None: #if expt is Random Confusion
        noise_expt = True

    name = args_re.name
    seed, device, dataset, lossfn, wt_decay = args_re.seed, 'cuda', args_re.dataset, args_re.lossfn, args_re.weight_decay
    arch, filters, forget_class, num_to_forget = args_re.model, 1., args_re.forget_class, args_re.num_to_forget
    MIA_info = {"num_classes":args.num_classes, "split":args.MIA_split, "seed":seed, "reps":args.MIA_reps, "RC":noise_expt}

    model = get_model(args_re.model, num_classes=args.num_classes).to('cuda')

    if args.custom:
        logger_cust = get_logger(folder=f'{args.result_folder_cust}/', logname=args.logname_cust)
        mkdir(f"{args.result_folder_cust}/{mat_folder}")
        mats_cust = f'{args.result_folder_cust}/{mat_folder}'
        model_cust = load_model(args.path_cust, copy.deepcopy(model), logger_cust)
        measure(logger_cust, model_cust, device, mats_cust, loader_re, loader_f, loader_te, 
                args.logname_cust, forget_class, args.exch_classes, MIA_info, loader_te_r, loader_te_f)
        exit(0)
    
    if not args.no_ogre:
        model_og, model_re = copy.deepcopy(model), copy.deepcopy(model)   
        if args.oldLoad:
            model_og.load_state_dict(torch.load(args.path_o))
            model_re.load_state_dict(torch.load(args.path_r))
        else:
            model_og = load_model(args.path_o, model_og, logger_un)
            model_re = load_model(args.path_r, model_re, logger_un)
        # Original
        weightpath = f"{prefixpath}/Weights"
        mkdir(weightpath)
        print_model_params(model_og, f'{weightpath}/Original.txt')
        mats_un = f'{prefixpath}/{mat_folder}'
        model_test(logger_un, model_og, device, mats_un, loader_re, loader_f, loader_te,
                    "Original", forget_class, args.exch_classes, MIA_info, loader_te_r, loader_te_f)
        model_test(logger_un, model_re, device, mats_un, loader_re, loader_f, loader_te, 
                    "Retrain", forget_class, args.exch_classes, MIA_info, loader_te_r, loader_te_f)

    # Get jacobian hessian stuff computed and stored
    if args.golatkar:
        mkdir(f"{prefixpath}/{args.name_go}/{mat_folder}")
        logger_go = get_logger(folder=f'{prefixpath}/{args.name_go}/', logname=logname)
        mats_go = f'{prefixpath}/{args.name_go}/{mat_folder}'

        #NTK
        model_ntk = load_model(args.path_ntk, copy.deepcopy(model), logger_go)
        measure(logger_go, model_ntk, device, mats_go, loader_re, loader_f, loader_te,
                "NTK", forget_class, args.exch_classes, MIA_info, loader_te_r, loader_te_f)

        #Fisher 
        modelf = load_model(args.path_fisher, copy.deepcopy(model), logger_go)
        measure(logger_go, modelf, device, mats_go, loader_re, loader_f, loader_te,
                "Fisher", forget_class, args.exch_classes, MIA_info, loader_te_r, loader_te_f)

        #NTK+Fisher stuff
        model_ntkf = load_model(args.path_ntkfisher, copy.deepcopy(model), logger_go)
        measure(logger_go, model_ntkf, device, mats_go, loader_re, loader_f, loader_te,
                "NTK_fisher", forget_class, args.exch_classes, MIA_info, loader_te_r, loader_te_f)

    #Finetune
    if args.finetune:
        logger_ft = get_logger(folder=f'{prefixpath}/{args.name_ft}/', logname=logname)
        mkdir(f"{prefixpath}/{args.name_ft}/{mat_folder}")
        mats_ft = f'{prefixpath}/{args.name_ft}/{mat_folder}'
        model_ft = load_model(args.path_ft, copy.deepcopy(model), logger_ft)
        measure(logger_ft, model_ft, device, mats_ft, loader_re, loader_f, loader_te, 
                "Finetune", forget_class, args.exch_classes, MIA_info, loader_te_r, loader_te_f)

    #Finetune last K layers
    if args.finetune_final:
        mkdir(f"{prefixpath}/{args.name_ftF}/{mat_folder}")
        logger_ftF = get_logger(folder=f'{prefixpath}/{args.name_ftF}/', logname=logname)
        weightpath = f"{prefixpath}/{args.name_ftF}/Weights"
        mkdir(weightpath)
        mats_ftF = f'{prefixpath}/{args.name_ftF}/{mat_folder}'

        for k in range(args.minL_ftF, args.maxL_ftF+1, args.stepL_ftF):
            model_ftF = load_model(f"{args.prefix_ftF}{k}.pt", copy.deepcopy(model), logger_ftF)
            print_model_params(model_ftF, f'{weightpath}/FTfinal{k}.txt')
            measure(logger_ftF, model_ftF, device, mats_ftF, loader_re, loader_f, loader_te,
                    f"FTfinal{k}", forget_class, args.exch_classes, MIA_info, loader_te_r, loader_te_f)
    
    #Retrain last K layers
    if args.retrfinal:
        mkdir(f"{prefixpath}/{args.name_rf}/{mat_folder}")
        logger_rf = get_logger(folder=f'{prefixpath}/{args.name_rf}/', logname=logname)
        weightpath = f"{prefixpath}/{args.name_rf}/Weights"
        mkdir(weightpath)
        mats_rf = f'{prefixpath}/{args.name_rf}/{mat_folder}'

        for k in range(args.minL_rf, args.maxL_rf+1, args.stepL_rf):
            model_rf = load_model(f"{args.prefix_rf}{k}.pt", copy.deepcopy(model), logger_rf)
            print_model_params(model_rf, f'{weightpath}/RetrFinal{k}.txt')
            measure(logger_rf, model_rf, device, mats_rf, loader_re, loader_f, loader_te,
                    f"RetrFinal{k}", forget_class, args.exch_classes, MIA_info, loader_te_r, loader_te_f)
