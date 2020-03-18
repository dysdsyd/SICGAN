import torch
import os
import shutil
import pandas as pd, random
import numpy as np
import logging
from tqdm import tqdm
import pickle

object_list = ['04379243',
 '02958343',
 '03001627',
 '02691156',
 '04256520',
 '04090263',
 '03636649',
 '04530566',
 '02828884',
 '03691459']



def train_val_split(config, ratio=0.7):
    '''
    Function for splitting dataset in train and validation
    '''
    print("Splitting Dataset..")
    data_dir = config.SHAPENET_DATA.PATH 
    taxonomy = pd.read_json(data_dir+'/taxonomy.json')
    classes = [i for i in os.listdir(data_dir) if i in '0'+taxonomy.synsetId.astype(str).values]
    random.shuffle(classes)
    assert classes != [], "No  objects(synsetId) found."
    ################ Pre-defined Classes #################
    classes = object_list
    ######################################################
    classes = dict(zip(classes,np.arange(len(classes))))
    
    ## Save the class to synsetID mapping
    path = os.path.join(config.CKP.experiment_path, 'class_map.pkl') 
    with open(path, 'wb') as handle:
        pickle.dump(classes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    trn_objs = []
    val_objs = []

    for cls in tqdm(classes):
        tmp = [(classes[cls], os.path.join(data_dir, cls, obj_file,'model.obj')) for obj_file in os.listdir(os.path.join(data_dir,cls))]
        random.shuffle(tmp)
        tmp_train = tmp[:int(len(tmp)*0.7)]
        tmp_test = tmp[int(len(tmp)*0.7):]
        trn_objs += tmp_train
        val_objs += tmp_test
        print(taxonomy['name'][taxonomy.synsetId == int(cls)], len(tmp))
    random.shuffle(trn_objs)
    random.shuffle(val_objs)
    return trn_objs, val_objs


def save_checkpoint(model, optimizer, curr_epoch, curr_step, args, curr_loss, curr_acc, trn_loss, filename ):
    """
        Saves a checkpoint and updates the best loss and best weighted accuracy
    """
    is_best_loss = curr_loss < args['best_loss']
    is_best_acc = curr_acc > args['best_acc']

    args['best_acc'] = max(args['best_acc'], curr_acc)
    args['best_loss'] = min(args['best_loss'], curr_loss)

    state = {   'epoch':curr_epoch,
                'step': curr_step,
                'args': args,
                'state_dict': model.state_dict(),
                'val_loss': curr_loss,
                'val_acc': curr_acc,
                'trn_loss':trn_loss,
                'best_val_loss': args['best_loss'],
                'best_val_acc': args['best_acc'],
                'optimizer' : optimizer.state_dict(),
             }
    path = os.path.join(args['experiment_path'], filename)
    torch.save(state, path)
    if is_best_loss:
        shutil.copyfile(path, os.path.join(args['experiment_path'], 'model_best_loss.pkl'))
    if is_best_acc:
        shutil.copyfile(path, os.path.join(args['experiment_path'], 'model_best_acc.pkl'))

    return args


def accuracy(output, target, topk=(1,)):
    """ From The PyTorch ImageNet example """
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def write_to_file(path, msg):
    f = open(os.path.join(path, 'out.txt'), 'a')
    f.write(msg)
    f.close()