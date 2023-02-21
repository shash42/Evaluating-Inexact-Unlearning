# pyright: reportMissingImports=true, reportUntypedBaseClass=false, reportGeneralTypeIssues=true
import torch.nn as nn
import copy
from models import _reinit
from learn import train_loop
from utils import load_model, load_pretrained


def getRetrainLayers(m, name, ret):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
        ret.append((m, name))
        #print(name)
    for child_name, child in m.named_children():
        getRetrainLayers(child, f'{name}.{child_name}', ret)
    return ret

def resetFinalResnet(model, num_retrain, modname, logger, reinit=True, pretrain_path=None):
    for param in model.parameters():
        param.requires_grad = False

    pre_ret = None
    if pretrain_path is not None:
        model_pre = copy.deepcopy(model)
        model_pre = load_pretrained(model_pre, modname, pretrain_path)
        pre_ret = getRetrainLayers(model_pre, 'M_pre', [])
        pre_ret.reverse()

    done = 0
    ret = getRetrainLayers(model, 'M', [])
    ret.reverse()
    for idx in range(len(ret)):
        if reinit:
            if isinstance(ret[idx][0], nn.Conv2d) or isinstance(ret[idx][0], nn.Linear):
                if pre_ret is not None:
                    ret[idx][0].weight, ret[idx][0].bias = pre_ret[idx][0].weight, pre_ret[idx][0].bias    
                    logger.info(f'Reinitialized layer: {ret[idx][1]}')
                else:
                    _reinit(ret[idx][0])
                    logger.info(f'Reinitialized layer: {ret[idx][1]}')
        if isinstance(ret[idx][0], nn.Conv2d) or isinstance(ret[idx][0], nn.Linear):
            done += 1
        for param in ret[idx][0].parameters():
            param.requires_grad = True
        if done >= num_retrain:
            break

    return model

MAX_LOSS = 1E8

def retrain_lastK(K, args, model, modname, retain_loader, valid_loader, test_loader, epochs, logger, prefixpath, pretrain_path=None):
    logger.info(f'Retraining last {K} layers!')
    model = resetFinalResnet(model, K, modname, logger, reinit=True, pretrain_path=pretrain_path) #Turns parameters to freeze off, reinitializes rest
    model = model.to(args.device)

    best_model, train_time = train_loop(model, args, epochs, logger, retain_loader, valid_loader, test_loader, prefixpath)
    return best_model, train_time

def cat_forget_finetune(args, model, modname, retain_loader, valid_loader, test_loader, epochs, logger, prefixpath, K=None):
    #If only partial model has to be finetuned:
    if K is not None:
        logger.info(f'Finetuning last {K} layers!')
        model = resetFinalResnet(model, K, modname, logger, reinit=False) #Turns parameters to freeze off, reinitializes rest if True
        model = model.to(args.device)
    else: logger.info(f'Finetuning entire model!')

    best_model, train_time = train_loop(model, args, epochs, logger, retain_loader, valid_loader, test_loader, prefixpath)
    return best_model, train_time
