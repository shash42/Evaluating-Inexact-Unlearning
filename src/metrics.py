# pyright: reportMissingImports=true, reportUntypedBaseClass=false, reportGeneralTypeIssues=true

from torch.functional import split
from utils import AverageMeter, get_error, loader_exch2_classes, mislabel_loader, seed_everything, test_loader_split
import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
from learn import test
from copy import deepcopy
from PrivacyRiskScore.MIA import black_box_benchmarks
from utils import subset_loader, split_loader

def get_outputs(model, loader):
    activations, predictions=[], []
    sample_info = []
    dataloader = torch.utils.data.DataLoader(loader.dataset, batch_size=1, shuffle=False)
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            outputs = model(inputs)
            curr_probs = torch.nn.functional.softmax(outputs,dim=1)
            curr_preds = torch.argmax(curr_probs, axis=1)
            activations.append(curr_probs.cpu().detach().numpy().squeeze())
            predictions.append(curr_preds.cpu().detach().numpy().squeeze())
            sample_info.append((inputs, outputs, labels))
    return np.stack(activations), np.array(predictions), sample_info

def get_metrics(model, dataloader, criterion):
    activations, predictions, sample_info = get_outputs(model, dataloader)
    losses, errors = AverageMeter(), AverageMeter()

    for inputs, outputs, labels in sample_info:
        loss = criterion(outputs, labels)
        losses.update(loss.item(), n=inputs.size(0))
        errors.update(get_error(outputs, labels), n=inputs.size(0))    
        
    return losses.avg, errors.avg, activations, predictions


def activations_predictions(logger, model, dataloader, name):
    criterion = torch.nn.CrossEntropyLoss()
    losses, errors, activations,predictions=get_metrics(model, dataloader, criterion)
    logger.info(f"{name} -> Loss:{np.round(losses,3)}, Error:{errors}")
    return activations, predictions

def predictions_distance(logger, l1,l2,name):
    dist = np.sum(np.abs(l1-l2))
    logger.info(f"Predictions Distance {name} -> {dist}")

def activations_distance(logger, a1,a2,name):
    dist = np.linalg.norm(a1-a2,ord=1,axis=1).mean()
    logger.info(f"Activations Distance {name} -> {dist}")

### Shash code to generate confusion matrix - Reference: https://deeplizard.com/learn/video/0LhiS6yu2qQ
@torch.no_grad()
def get_all_preds(model, loader, device):
    all_preds, all_targets = torch.tensor([]), torch.tensor([])
    all_preds, all_targets = all_preds.to(device), all_targets.to(device)
    for batch in loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        model.eval()
        preds = model(images)
        all_preds = torch.cat((all_preds, preds),dim=0)
        all_targets = torch.cat((all_targets, labels), dim=0)
    return all_preds, all_targets

def classpairs_sorted_by_conf(cm):
    '''
    Returns list of 5-tuple sorted in ascending order of total confusion:
    (cm[i, j] + cm[j, i], i, j, cm[i, j], cm[j, i])
    '''
    pair_confs = []
    for i in range(cm.shape[0]):
        for j in range(i+1, cm.shape[1]):
            pair_confs.append((cm[i, j] + cm[j, i], i, j, cm[i, j], cm[j, i]))
    pair_confs.sort(reverse=True)
    return pair_confs

def gen_confu_mat(logger, model, loader, device, title = 'Confusion Matrix', path=None):
    preds, targets = get_all_preds(model, loader, device)
    #print(preds.argmax(dim=1))
    #print(targets)
    
    stacked = torch.stack((targets, preds.argmax(dim=1)), dim=1) 
    cmt = torch.zeros(preds.shape[1],preds.shape[1], dtype=torch.int64)
    for p in stacked:
        tl, pl = p.tolist()
        tl, pl = int(tl), int(pl)
        cmt[tl, pl] = cmt[tl, pl] + 1
    cm = np.array(cmt)
    #logger.debug(cm)
    
    if cm.shape[0] >= 5:
        pair_confs = classpairs_sorted_by_conf(cm)
        with open(f"{path}.txt", "w") as f:
            for pair in pair_confs:
                print(f'Classes ({pair[1]}, {pair[2]}) - {pair[3]} + {pair[4]} = {pair[0]}', file=f)
            for i in range(cm.shape[0]):
                print(f'Class {i} correct: {cm[i, i]}', file=f)
        return cm

    cmap=plt.cm.Blues
    classes = [i for i in range(preds.shape[1])]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'# if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if path is not None:
        plt.savefig(f"{path}.png")
        
    #plt.figure(figsize=(7,7))
    #plt.plot()
    plt.close()
    return cm

def CRT_score(cm, forget_class):
    FP = cm[forget_class][forget_class]
    FN = 0
    for i in range(cm.shape[0]):
        if i == forget_class:
            continue
        FN += cm[i][forget_class]
    return FP, FN

def Conf_Score(cm, classes):
    confscore = 0
    for i in range(len(classes)):
        for j in range(i+1, len(classes)):
            ci, cj = classes[i], classes[j]
            confscore += cm[ci][cj] + cm[cj][ci]
    return confscore

def acc_from_cmtx(cm):
    corr, tot = 0, 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            tot += cm[i][j]
            if i == j:
                corr += cm[i][j]
    return float(corr)/tot

def prepareMIAInputs(model, loader, split_ratio):
    shadow_loader, target_loader, _, _ = split_loader(loader, split_ratio)
    
    shadow_un_out, _, _ = get_outputs(model, shadow_loader)
    target_un_out, _, _ = get_outputs(model, target_loader)

    shadow_perf = (shadow_un_out, shadow_loader.dataset.targets)
    target_perf = (target_un_out, target_loader.dataset.targets)
    return shadow_perf, target_perf

def SongMIA(logger, model, num_classes, forget_loader, test_loader,
             split_ratio, repetitions, seed, exch=None, rc=False):
    '''
    Special case implemented for 2 class confusion in which all forget, t_f samples passed to MIA 
    '''
    #Ensure benchmarks keys are same as those returned by ._mia_inf_benchmarks()
    benchmarks = {"corr_sha":0, "corr_tar":0, "conf_sha":0, "conf_tar":0,
                        "entr_sha":0, "entr_tar":0, "mentr_sha":0, "mentr_tar":0}
    seed_everything(seed)

    if exch is not None and len(exch)==2:
        forget_loader = loader_exch2_classes(forget_loader, exch)
        test_loader = loader_exch2_classes(test_loader, exch)
        
    forget_is_smaller = True
    if len(test_loader.dataset) < len(forget_loader.dataset):
        forget_is_smaller = False
    
    if forget_is_smaller:
        shadow_un_perf, target_un_perf = prepareMIAInputs(model, forget_loader, split_ratio)
    else:
        shadow_te_perf, target_te_perf = prepareMIAInputs(model, test_loader, split_ratio)
    
    for _ in range(repetitions):
        if forget_is_smaller:
            test_loader, _ = subset_loader(test_loader, len(forget_loader.dataset)) #To ensure balanced categories
            shadow_te_perf, target_te_perf = prepareMIAInputs(model, test_loader, split_ratio)
        else:
            forget_loader, _ = subset_loader(forget_loader, len(test_loader.dataset)) #To ensure balanced categories
            shadow_un_perf, target_un_perf = prepareMIAInputs(model, forget_loader, split_ratio)

        if rc:
            shadow_te_perf, shadow_un_perf = shadow_un_perf, shadow_te_perf
            target_te_perf, target_un_perf = target_un_perf, target_te_perf
            logger.info(f'Swapped un, te for RC test')

        MIA_obj = black_box_benchmarks(shadow_un_perf, shadow_te_perf, target_un_perf, target_te_perf, num_classes)
        benchmarks_curr = MIA_obj._mem_inf_benchmarks(all_methods=False, benchmark_methods=['confidence'])
        for key, value in benchmarks_curr.items():
            if key in benchmarks.keys():
                benchmarks[key] += value
    
    for key in benchmarks.keys():
        benchmarks[key] /= repetitions
    conf = 'Exch-' if exch is not None else ''
    logger.info(f'{conf}MIA-Correct - Shadow: {benchmarks["corr_sha"]}\tTarget: {benchmarks["corr_tar"]}')
    logger.info(f'{conf}MIA-Confidence - Shadow: {benchmarks["conf_sha"]}\tTarget: {benchmarks["conf_tar"]}')
    logger.info(f'{conf}MIA-Entropy - Shadow: {benchmarks["entr_sha"]}\tTarget: {benchmarks["entr_tar"]}')
    logger.info(f'{conf}MIA-Mod.Entropy - Shadow: {benchmarks["mentr_sha"]}\tTarget: {benchmarks["mentr_tar"]}')


def model_test(logger, model, device, savepath, retain_loader, forget_loader, test_loader_full, 
                name, forget_class, exch_classes, MIA_info=None, test_loader_r=None, test_loader_f=None):
    #Perform MIA
    if MIA_info is not None:
        #Classify whether outputs are for a sample in D_f or D_te_f. If D_te_f is None, takes D_te.
        num_classes, split, seed, reps, rc =\
             MIA_info["num_classes"], MIA_info["split"], MIA_info["seed"], MIA_info["reps"], MIA_info["RC"]
        if test_loader_f is None: #For example in noise experiments
            SongMIA(logger, model, num_classes, deepcopy(forget_loader), deepcopy(test_loader_full), split, reps, seed, rc=rc)
        elif exch_classes is not None and len(exch_classes) == 2:
            # SongMIA(logger, model, num_classes, deepcopy(forget_loader), deepcopy(test_loader_f), split, reps, seed)
            SongMIA(logger, model, num_classes, deepcopy(forget_loader), deepcopy(test_loader_f), split, reps, seed, exch=exch_classes)
        else:            
            SongMIA(logger, model, num_classes, deepcopy(forget_loader), deepcopy(test_loader_f), split, reps, seed)
        
    #Prints losses and errors to logger        
    activations_r, predictions_r = activations_predictions(logger, model, retain_loader, 
                                                                                        f'{name}_Model_D_r')
    activations_f, predictions_f = activations_predictions(logger, model, forget_loader, 
                                                                                        f'{name}_Model_D_f')
    activations_t, predictions_t = activations_predictions(logger, model,test_loader_full, 
                                                                                        f'{name}_Model_D_t')
    if test_loader_r is not None:
        activations_te_r, predictions_te_r = activations_predictions(logger, model,test_loader_r, 
                                                                                        f'{name}_Model_D_te_r')
    if test_loader_f is not None:
        activations_te_f, predictions_te_f = activations_predictions(logger, model,test_loader_f, 
                                                                                        f'{name}_Model_D_te_f')
    #Make confusion matrices and get confusion scores
    CRTRes, ConfRes = {}, {}
    cm_t = gen_confu_mat(logger, model, test_loader_full, device, f"{name} - Test - Confusion Matrix", f"{savepath}/{name}-test")
    cm_f = gen_confu_mat(logger, model, forget_loader, device, f"{name} - Forget - Confusion Matrix", f"{savepath}/{name}-forget")
    cm_r = gen_confu_mat(logger, model, retain_loader, device, f"{name} - Retain - Confusion Matrix", f"{savepath}/{name}-retain")
    cm_te_r, cm_te_f = None, None
    if test_loader_r is not None:
        cm_te_r = gen_confu_mat(logger, model, test_loader_r, device, f"{name} - Test_r - Confusion Matrix", f"{savepath}/{name}-test_r")
    if test_loader_f is not None:
        cm_te_f = gen_confu_mat(logger, model, test_loader_f, device, f"{name} - Test_f - Confusion Matrix", f"{savepath}/{name}-test_f")

    if forget_class is not None and forget_class != -1:
        logger.info(f'Class Forgetting performance of {name}')
        CRTRes["t_FP"], CRTRes["t_FN"] = CRT_score(cm_t, forget_class)
        logger.info(f'Test:\tt_FP = {CRTRes["t_FP"]}\tt_FN = {CRTRes["t_FN"]}')
        CRTRes["r_FP"], CRTRes["r_FN"] = CRT_score(cm_r, forget_class)
        logger.info(f'Retain:\tr_FP = {CRTRes["r_FP"]}\tr_FN = {CRTRes["r_FN"]}')
        CRTRes["f_FP"], CRTRes["f_FN"] = CRT_score(cm_f, forget_class)
        logger.info(f'Forget:\tf_FP = {CRTRes["f_FP"]}\tf_FN = {CRTRes["f_FN"]}')
        if cm_te_r is not None:
            CRTRes["te_r_FP"], CRTRes["te_r_FN"] = CRT_score(cm_te_r, forget_class)
            logger.info(f'Test_r:\tte_r_FP = {CRTRes["te_r_FP"]}\tte_r_FN = {CRTRes["te_r_FN"]}')
        if cm_te_f is not None:
            CRTRes["te_f_FP"], CRTRes["te_f_FN"] = CRT_score(cm_te_f, forget_class)
            logger.info(f'Test_f:\tte_f_FP = {CRTRes["te_f_FP"]}\tte_f_FN = {CRTRes["te_f_FN"]}')

    logger.debug(f'Exch classes - {exch_classes}')
    if exch_classes is not None:
        logger.info(f'Confusion Forgetting performance of {name}')
        ConfRes["t_Conf"] = Conf_Score(cm_t, exch_classes)
        logger.info(f'Test:\tConf_Score = {ConfRes["t_Conf"]}')
        ConfRes["f_Conf"] = Conf_Score(cm_f, exch_classes)  
        logger.info(f'Forget:\tConf_Score = {ConfRes["f_Conf"]}')
        ConfRes["r_Conf"] = Conf_Score(cm_r, exch_classes)
        logger.info(f'Retain:\tConf_Score = {ConfRes["r_Conf"]}')


def output_distance(logger, activations, predictions, m_short1, m_short2, name):
    predictions_distance(logger, predictions[f"{m_short1}_D_f"], predictions[f"{m_short2}_D_f"],f'{name}_D_f')
    activations_distance(logger, activations[f"{m_short1}_D_f"], activations[f"{m_short2}_D_f"],f'{name}_D_f')
    activations_distance(logger, activations[f"{m_short1}_D_r"], activations[f"{m_short2}_D_r"],f'{name}_D_r')
    activations_distance(logger, activations[f"{m_short1}_D_t"], activations[f"{m_short2}_D_t"],f'{name}_D_t')

''' REWRITE COMPLETELY
def CRTvsEpochs(args, model_r, epochs, retain_loader, test_loader, num_classes, forget_class, logger):
    Acc, Tfp, Tfn = [], [], []
    for epoch in range(epochs):
        model_r = retrain_last(args, model_r, retain_loader, test_loader, 1, num_classes, 0.01, logger, reset_last=(epoch==0))
        cm_t = gen_confu_mat(args, model_r, test_loader, None)
        t_FP, t_FN = CRT_score(cm_t, forget_class)
        acc = acc_from_cmtx(cm_t)
        Acc.append(acc)
        Tfp.append(t_FP)
        Tfn.append(t_FN)
    return Acc, Tfp, Tfn
'''

