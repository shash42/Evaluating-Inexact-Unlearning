# pyright: reportMissingImports=true, reportUntypedBaseClass=true, reportGeneralTypeIssues=true
#TODO: Removal all PRINTS
import os, time, argparse, copy, pickle, torch #toch only required for: torch.load
from load_datasets import get_loaders
from learn import caller, learn
from metrics import model_test, output_distance
from golatkar import golatkar_precomputation, get_ntk_model, fisher_init, \
                    apply_fisher_noise
from membership import membership_attack
from methods import retrain_lastK, cat_forget_finetune
from utils import load_model, print_model_params, save_model, get_forget_retain_loader, print_final_times, get_logger
from models import get_model

def parse_args():
    # Training settings - add confusion(?), cutmix_alpha, log_dir, exp_name
    parser = argparse.ArgumentParser()
    parser.add_argument('--augment', action='store_true', default=False,
                        help='Use data augmentation')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--disable-bn', action='store_true', default=False,
                        help='Put batchnorm in eval mode and don\'t update the running averages')
    parser.add_argument('--epochs', type=int, default=31, metavar='N',
                        help='number of epochs to train (default: 31)')
    parser.add_argument('--filters', type=float, default=1.0,
                        help='Percentage of filters')
    parser.add_argument('--forget-class', type=int, default=None,
                        help='Class to forget')
    parser.add_argument('--l1', action='store_true', default=False,
                        help='uses L1 regularizer instead of L2')
    parser.add_argument('--lossfn', type=str, default='ce',
                        help='Cross Entropy: ce or mse')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--model', default='mlp')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of Classes')
    parser.add_argument('--num-to-forget', type=int, default=None,
                        help='Number of samples of class to forget')
    parser.add_argument('--name', default=None)
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint to resume')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--step-size', default=None, type=int, help='learning rate scheduler')
    parser.add_argument('--unfreeze-start', default=None, type=str, help='All layers are freezed except the final layers starting from unfreeze-start')
    parser.add_argument('--weight-decay', type=float, default=0.0005, metavar='M',
                        help='Weight decay (default: 0.0005)')
    args = parser.parse_args()
    return args

def measure(logger, model, device, ConfMatsPath, retain_loader, forget_loader, test_loader_full, 
            met_fname, met_sname, activations, predictions, forget_class, confusion_eval):
    model_test(logger, model, device, ConfMatsPath, retain_loader, forget_loader, test_loader_full,
                                                     met_fname, forget_class, confusion_eval)
    #output_distance(logger, activations, predictions, ogmod_name, met_sname, f"Retrain_{met_fname}")

def pretrain(mod_name, exp_name, log_dir):
    args_dict = {"dataset":"cifar100", "model":mod_name, "filters":1.0, "lossfn":"ce",
             "num_classes":100, "batch_size":64, "log_dir":log_dir, "exp_name":exp_name,
             "procedure":"pretrain", "epochs":126}
    args, logger, train_loader, valid_loader, test_loader, confu_idxs, model, epoch = caller(args_dict)
    return args, logger, train_loader, valid_loader, test_loader, model, epoch

def finetune(mod_name, exp_name, log_dir, resumePath, epochs=62, confusion = None, confusion_copy=False, confname=None):
    args_dict = {"dataset":"small_cifar5", "model":mod_name, "filters":1.0, "maxlr":0.01, 
             "resume":resumePath, "log_dir":log_dir, "exp_name":exp_name, "procedure":"original",
             "disable_bn":False, "epochs":epochs, "seed":1, "batch_size":64, "weight_decay":5e-5, #ideally: 5e-5, golat 0.1
             "confname":confname, "confusion":confusion,"confusion_copy":confusion_copy}
    args, logger, train_loader, valid_loader, test_loader, confu_idxs, model, epoch = caller(args_dict)
    return args, logger, train_loader, valid_loader, test_loader, confu_idxs, model, epoch

#ENSURE SAME SEED AS FINETUNE
def retrain_from_scratch(mod_name, exp_name, log_dir, resumePath, epochs=62, C_f=None, N_f=0, confusion_indices=None):
    args_dict = {"dataset":"small_cifar5", "model":mod_name, "filters":1.0, "maxlr":0.01, 
                "weight_decay":5e-5, "disable_bn":False, "epochs":epochs, "seed":1, "batch_size":64, #ideally: decay 5e-5
             "resume":resumePath, "log_dir":log_dir, "exp_name":exp_name, "procedure":"retrain",
             "forget_class":C_f, "num_to_forget":N_f, "forget_indices":confusion_indices}
    args, logger, train_loader, valid_loader, test_loader, confu_idxs, model, epoch = caller(args_dict)
    return args, logger, train_loader, valid_loader, test_loader, model, epoch

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
                if i != j:  mat[i][j] = num_change

    if conftype == 'exchange':
        assert(exch_classes is not None)
        for i in exch_classes:
            for j in exch_classes:
                if i != j:  mat[i][j] = num_change

    return mat

if __name__ == "__main__":
    #Model pretrain, finetune, and obtain retrained. Make changes in these functions for diff. types of models
    #args = parse_args()

    ##########################################################################################################
    #Arguments

    #Naming and other Necessary Stuff
    logname = 'first-checkpoint'
    mod_name = "resnet20"
    log_dir = "logs/finetune-MaxLR1MinLR1e-2-debug"
    exp_name = "30eps-TrainAugON-Conf-C0-C1-10"
    ResultName = "Matrices"
    num_classes = 5

    #Class Removal Stuff
    C_f = 0       #Pass C_f, N_f if forgetting from single class test, none otherwise.
    N_f = 100

    #Confusion Stuff
    confname = "C0-C1-10" #Set to confusion test name if using confusion test, else None
    conftype = 'exchange'
    num_change = 10
    exch_classes = [0, 1] #List containing classes to be confused eg [0, 1] (C_{i,j}=num_change for all i, j)

    #Other metrics/methods stuff
    ep_orig = 62
    retrfinal_epochs, retrfinal_L, retrfinal_R, retrfinal_mult = 62, 1, 20, 0.01
    finetune_epochs = 30

    #TODO: Switch back to 62ep pretrained model
    path_pre = f'logs/128ep-Pretrain-Resnet20-SmallCifar5/CRT-C0-All'\
               f'/cifar100_resnet20_1_0_forget_None_confused_None_num_0_maxlr_0_05_bs_64_wd_0_0005_seed_1_pretrain.pt'
    #path_pre = None

    #NEED:
    #model_g we just need architecture, should be easy to make same.
    #loaders should be easy to get, just make the same call with same seed.
    #args dump somewhere from OG and pick them up :D
    oldLoad = False #Do set transform_tr to False
    if oldLoad:
        path_o = 'logs/golatkar-fixing/checkpoints/original-Aug-Transf-exch-c0-c1-10-golatkar_25.pt'
        path_r = 'logs/golatkar-fixing/checkpoints/retrain-Aug-Transf-exch-c0-c1-10-golatkar_25.pt'
        path_oarg = 'logs/golatkar-fixing/notebook_dumps/AugOn-Transf-Exch-C0-C1-10/args_og.txt'
        path_rarg = 'logs/golatkar-fixing/notebook_dumps/AugOn-Transf-Exch-C0-C1-10/args_re.txt'
        init_checkpoint = 'logs/golatkar-fixing/checkpoints/retrain-Aug-Transf-exch-c0-c1-10-golatkar_init.pt'
    else:
        path_o, path_oarg = None, None
        path_r, path_rarg, init_checkpoint = None, None, None

    #########################################################################################################
    #Beyond this point only edit call parameters to finetune and retrain_from_scratch

    ConfMatsPath = f"{log_dir}/{exp_name}/{ResultName}"
    if not os.path.isdir(f"{log_dir}"):
        os.mkdir(f"{log_dir}")
    if not os.path.isdir(f"{log_dir}/{exp_name}"):
        os.mkdir(f"{log_dir}/{exp_name}")
    if not os.path.isdir(ConfMatsPath):
        os.mkdir(ConfMatsPath)

    logger = get_logger(folder=log_dir+'/'+exp_name+'/', logname=logname)

    if path_pre is None:
        args_pre, logger, train_loader, valid_loader, test_loader, model, epoch = pretrain(mod_name, exp_name, log_dir) 
        path_pre = f'{args_pre.log_dir}/{args_pre.exp_name}/{args_pre.name}_{args_pre.procedure}.pt'
        
    confusion, confusion_eval = None, None
    if confname is not None:
        confusion = confmat(num_classes, conftype, num_change, exch_classes)
        if conftype == 'exchange':
            confusion_eval = exch_classes

    compute = 0
    if path_o is None:
        starttime = time.monotonic()
        args_og, logger, train_loader_full, valid_loader_full, test_loader_full, confu_idxs, model_o, epoch\
            = finetune(mod_name, exp_name, log_dir, path_pre, epochs=ep_orig, confusion=confusion, confname=confname)
        compute = time.monotonic() - starttime
        path_o = f'{args_og.log_dir}/{args_og.exp_name}/{args_og.name}_{args_og.procedure}.pt'
        if confname is not None:  
            C_f = None
            N_f = args_og.num_to_forget
        if not os.path.isdir(f"{log_dir}/{exp_name}"):
            os.mkdir(f"{log_dir}/{exp_name}")
        path_oarg = f"{log_dir}/{exp_name}/args_og.txt"
        pickle.dump(args_og, open(path_oarg, 'wb'))

    compute0 = 0
    if path_r is None:
        starttime = time.monotonic()
        args_re, logger, train_loader_g, valid_loader_g, test_loader_g,\
            model_g, epoch = retrain_from_scratch(mod_name, exp_name, log_dir, path_pre, epochs=ep_orig, confusion_indices=confu_idxs, C_f=C_f, N_f=N_f)
        compute0 = time.monotonic() - starttime
        path_r = f'{args_re.log_dir}/{args_re.exp_name}/{args_re.name}_{args_re.procedure}.pt'
        init_checkpoint = f"{args_re.log_dir}/{args_re.exp_name}/{args_re.name}_init.pt"
        if not os.path.isdir(f"{log_dir}/{exp_name}"):
            os.mkdir(f"{log_dir}/{exp_name}")
        path_rarg = f"{log_dir}/{exp_name}/args_re.txt"
        pickle.dump(args_re, open(path_rarg, 'wb'))

    args_og = pickle.load(open(path_oarg, 'rb'))
    args_re = pickle.load(open(path_rarg, 'rb'))
    

    logger.info(f'Original loader extraction\n C_f: {args_og.forget_class}\n N_f: {args_og.num_to_forget}\n idxrep: {args_og.forget_indices}\n'
    f'seed: {args_og.seed}\n confu_idxs: {args_og.confusion}\n batch: {args_og.batch_size}\n data: {args_og.dataset}')
    train_loader_full, valid_loader_full, test_loader_full, confu_idxs = \
                                                    get_loaders(logger, args_og.dataset, class_to_replace=args_og.forget_class,
                                                     num_indexes_to_replace=args_og.num_to_forget, 
                                                     indexes_to_replace=args_og.forget_indices, batch_size=args_og.batch_size, 
                                                     seed=args_og.seed, confusion=args_og.confusion, confusion_copy=args_og.confusion_copy,
                                                     transform_tr = True)

    logger.info(f'Retain/Forget extraction\n C_f: {args_re.forget_class}\n N_f: {args_re.num_to_forget}\n seed: {args_re.seed}\n confu_idxs: {confu_idxs}\n batch: {args_re.batch_size}')
    marked_loader, _, _, _ = get_loaders(logger, args_re.dataset, class_to_replace=args_re.forget_class, num_indexes_to_replace=args_re.num_to_forget,
                                                 only_mark=True, batch_size=1, seed=args_re.seed, shuffle=True,
                                                 indexes_to_replace=confu_idxs, transform_tr = True)
    forget_loader, retain_loader = get_forget_retain_loader(marked_loader, train_loader_full, args_re.seed, args_re.batch_size)
    logger.debug(f'retain targets: {retain_loader.dataset.targets}')
    num_total = len(train_loader_full.dataset)
    num_to_retain = len(retain_loader.dataset)
    num_to_forget = num_total - num_to_retain

    name = args_re.name
    seed, device, dataset, lossfn, wt_decay = args_re.seed, 'cuda', args_re.dataset, args_re.lossfn, args_re.weight_decay
    arch, filters, forget_class, num_to_forget = args_re.model, args_re.filters, args_re.forget_class, args_re.num_to_forget
    
    # retrfinal_args = argparse.Namespace(device = device, log_dir = log_dir, exp_name = exp_name, name = 'retrfinal',
    #                             momentum=args_re.momentum, disable_bn = args_re.disable_bn, regularization = args_re.regularization, 
    #                             cutmix_prob = args_re.cutmix_prob, cutmix_alpha = args_re.cutmix_alpha,
    #                             clip = args_re.clip, weight_decay = args_re.weight_decay,
    #                             maxlr = args_re.maxlr, minlr = args_re.minlr)
    finetune_args = argparse.Namespace(device = device, log_dir = log_dir, exp_name = exp_name, name = 'ft_baseline',
                                momentum=args_re.momentum, disable_bn = args_re.disable_bn, regularization = args_re.regularization, 
                                cutmix_prob = args_re.cutmix_prob, cutmix_alpha = args_re.cutmix_alpha,
                                clip = args_re.clip, weight_decay = args_re.weight_decay,
                                maxlr = 1, minlr = 1e-2) #Maxlr set to pretrain one (0.05)

    #args used ends here! MODEL EVALUATION BEGINS!
    #################################################################################################################

    #model is original, model0 is retrain-from-scratch 
    model_g = get_model(args_re.model, num_classes=num_classes, filters_percentage=args_re.filters).to('cuda')
    model, model0 = copy.deepcopy(model_g), copy.deepcopy(model_g)   
    print(path_o, path_r)
    if oldLoad:
        model.load_state_dict(torch.load(path_o))
        model0.load_state_dict(torch.load(path_r))
    else:
        model = load_model(path_o, model, logger)
        model0 = load_model(path_r, model0, logger)

    #'''
    #Get jacobian hessian stuff computed and stored
    # logger.info(f'lossfn: {lossfn}, wt_dec: {wt_decay}\n chkpt: {init_checkpoint},'\
    #     f'#tot: {num_total}, #ret: {num_to_retain}, filters: {filters}')
    # starttime = time.monotonic() 
    # model_init, scale_ratio, delta_w, delta_w_actual, w_retain, predicted_scale = golatkar_precomputation(
    #     model, model0, retain_loader, forget_loader, seed, device, dataset, lossfn, wt_decay,
    #     init_checkpoint, arch, num_classes, num_total, num_to_retain, filters, "/scratch/NTK_data")
    # precompute_golatkar = time.monotonic() - starttime
    #'''

    # Original
    with open('logs/golatkar-fixing/weightsbeforeeval.txt', 'w') as f:
        for name, param in model.named_parameters():
            print(name, param.data, file=f)
    activations, predictions = {}, {}
    model_test(logger, model, device, ConfMatsPath, retain_loader, forget_loader, test_loader_full,
                                                     "Original", forget_class, confusion_eval)
    model_test(logger, model0, device, ConfMatsPath, retain_loader, forget_loader, test_loader_full,
                                                     "Retrain", forget_class, confusion_eval)
    #output_distance(logger, activations, predictions, "m0", "m", "Retrain_Original")
    
    #'''
    #NTK
    #model_scrub in notebook
    # starttime = time.monotonic()
    # model_ntk = get_ntk_model(device, model, predicted_scale, delta_w)
    # compute_ntk = time.monotonic() - starttime
    # measure(logger, model_ntk, device, ConfMatsPath, retain_loader, forget_loader, test_loader_full,
    #         "m0", "NTK", "ntk", activations, predictions, seed, forget_class, confusion_eval)

    # #Fisher stuff
    # starttime = time.monotonic()
    # modelf = fisher_init(device, retain_loader.dataset, model)
    # modelf0 = fisher_init(device, retain_loader.dataset, model0)
    # apply_fisher_noise(seed, num_classes, num_to_forget, modelf, modelf0)
    # compute_f = time.monotonic() - starttime
    # measure(logger, modelf, device, ConfMatsPath, retain_loader, forget_loader, test_loader_full,
    #         "Fisher", "fisher", activations, predictions, forget_class, confusion_eval)

    # #NTK+Fisher stuff
    # starttime = time.monotonic()
    # model_ntkf = fisher_init(device, retain_loader.dataset, model_ntk)
    # modelf0 = fisher_init(device, retain_loader.dataset, model0)
    # apply_fisher_noise(seed, num_classes, num_to_forget, model_ntkf, modelf0)
    # compute_ntkf = time.monotonic() - starttime
    # measure(logger, model_ntkf, device, ConfMatsPath, retain_loader, forget_loader, test_loader_full,
    #         "NTK_fisher", "ntk_fisher", activations, predictions, forget_class, confusion_eval)

    #Finetune
    starttime = time.monotonic()
    model_ft = copy.deepcopy(model)
    #retain_loader = replace_loader_dataset(train_loader_full,retain_dataset, seed=seed, batch_size=args.batch_size, shuffle=True)    
    model_ft = cat_forget_finetune(finetune_args, model_ft, retain_loader, valid_loader_full, test_loader_full, finetune_epochs, logger)
    compute_ft = time.monotonic() - starttime
    measure(logger, model_ft, device, ConfMatsPath, retain_loader, forget_loader, test_loader_full,
            "Finetune", "finetune", activations, predictions, forget_class, confusion_eval)

    #Retrain last K layers
    
    # retrfinal_models, retrfinal_times = [], []
    # for k in range(retrfinal_L, retrfinal_R+1):
    #     starttime = time.monotonic()
    #     model_r = copy.deepcopy(model)
    #     retrfinal_args.name = f'RetrFinal_{k}.pt'
    #     model_r = retrain_lastK(k, retrfinal_mult, retrfinal_args, model_r, retain_loader, valid_loader_full, test_loader_full, retrfinal_epochs, logger)
    #     compute_r = time.monotonic() - starttime
    #     retrfinal_models.append(model_r)
    #     retrfinal_times.append(compute_r)

    # if not os.path.isdir(f"{ConfMatsPath}/Retr"):
    #     os.mkdir(f"{ConfMatsPath}/Retr")
    # if not os.path.isdir(f"{log_dir}/{exp_name}/Weights"):
    #     os.mkdir(f"{log_dir}/{exp_name}/Weights")    
    # print_model_params(model, f'{log_dir}/{exp_name}/Weights/Original.txt')
    # for k in range(retrfinal_L, retrfinal_R+1):
    #     logger.info(f'Retrained last {k} layers. Time: {retrfinal_times[k - retrfinal_L]}')
    #     model_r = retrfinal_models[k - retrfinal_L]
    #     print_model_params(model_r, f'{log_dir}/{exp_name}/Weights/RetrFinal{k}.txt')
    #     measure(logger, model_r, device, ConfMatsPath, retain_loader, forget_loader, test_loader_full,
    #             f"Retr/Final{k}", f"retr{k}", activations, predictions, forget_class, confusion_eval)
  
    #Acc_r, TFP_r, TFN_r = CRTvsEpochs(model_ret, 10, retain_loader, test_loader_full, num_classes, forget_class, logger)
    #logger.info(f'Accuracy over epochs: {Acc_r}')
    #logger.info(f'Test FP over epochs: {TFP_r}')
    #logger.info(f'Test FN over epochs: {TFN_r}')

    #Membership Attack
    '''
    attack_dict = {}
    attack_dict['Original']=membership_attack(logger, retain_loader,forget_loader,test_loader_full,model, "Original")
    attack_dict['Retrain']=membership_attack(logger, retain_loader,forget_loader,test_loader_full,model0, "Retrain")
    attack_dict['NTK']=membership_attack(logger, retain_loader,forget_loader,test_loader_full,model_ntk, "NTK")
    attack_dict['Fisher']=membership_attack(logger, retain_loader,forget_loader,test_loader_full,modelf, "Fisher")
    attack_dict['Fisher_NTK']=membership_attack(logger, retain_loader,forget_loader,test_loader_full,model_ntkf, "NTKFisher")
    # attack_dict['Finetune']=membership_attack(logger, retain_loader,forget_loader,test_loader_full,model_ft, "Finetune")
    # attack_dict['RetrainLast']=membership_attack(logger, retain_loader,forget_loader,test_loader_full,model_r, "RetrainLast")
    '''

    #print_final_times(logger, precompute_golatkar, compute, compute0, compute_ntk, compute_f, compute_ntkf, compute_ft, compute_r)
    #print_final_times(logger, precompute_golatkar, compute, compute0, compute_ntk, compute_f, compute_ntkf, 0, 0)
    #LEFT - Information in activations, Information vs test error plot, calls of all_readouts