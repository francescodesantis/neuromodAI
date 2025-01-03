import argparse

from utils import load_presets, get_device, load_config_dataset, seed_init_fn, str2bool
from model import load_layers
from train import run_sup, run_unsup, check_dimension, training_config, run_hybrid, evaluate_sup, evaluate_unsup
from log_m import Log, save_logs
from dataset import make_data_loaders
from engine_cl import getActivation
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Multi layer Hebbian Training')

parser.add_argument('--preset', choices=load_presets(), default=None,
                    type=str, help='Preset of hyper-parameters ' +
                                   ' | '.join(load_presets()) +
                                   ' (default: None)')

parser.add_argument('--dataset-unsup', choices=load_config_dataset(), default='MNIST',
                    type=str, help='Dataset possibilities ' +
                                   ' | '.join(load_config_dataset()) +
                                   ' (default: MNIST)')

parser.add_argument('--dataset-sup', choices=load_config_dataset(), default='MNIST',
                    type=str, help='Dataset possibilities ' +
                                   ' | '.join(load_config_dataset()) +
                                   ' (default: MNIST)')

parser.add_argument('--training-mode', choices=['successive', 'consecutive', 'simultaneous'], default='successive',
                    type=str, help='Training possibilities ' +
                                   ' | '.join(['successive', 'consecutive', 'simultaneous']) +
                                   ' (default: successive)')

parser.add_argument('--resume', choices=[None, "all", "without_classifier"], default=None,
                    type=str, help='Resume Model ' +
                                   ' | '.join(["best", "last"]) +
                                   ' (default: None)')

parser.add_argument('--model-name', default=None, type=str, help='Model Name')

parser.add_argument('--training-blocks', default=None, nargs='+', type=int,
                    help='Selection of the blocks that will be trained')

parser.add_argument('--seed', default=None, type=int,
                    help='Selection of the blocks that will be trained')

parser.add_argument('--gpu-id', default=0, type=int, metavar='N',
                    help='Id of gpu selected for training (default: 0)')

parser.add_argument('--save', default=True, type=str2bool, metavar='N',
                    help='')

parser.add_argument('--validation', default=False, type=str2bool, metavar='N',
                    help='')

parser.add_argument('--evaluate', default=False, type=str2bool, metavar='N',
                    help='')

import gc



feature_maps = {}

def main(blocks, name_model, resume, save, evaluate, dataset_sup_config, dataset_unsup_config, train_config, gpu_id):
    device = get_device(gpu_id)
    model = load_layers(blocks, name_model, resume)

    model = model.to(device)

    depth = 0

    # here we obtain the activations of all the layers (which are convolutional layers)
    for layer in model.children():
        print(list(model.children()))
        print("LAYER NAME: " , layer)
        print("LAYER CHILDREN: " , list(layer.children()))
	# check for convolutional layer
        for subl in layer.children():
            if not subl.__eq__(None):
                print("SUBLAYER NAME: " , subl)
            for subsubl in subl.children():
                print("subsubl NAME: " , subsubl)
                if subsubl._get_name().__eq__("HebbSoftKrotovConv2d"):
                    subsubl.register_forward_hook(getActivation("conv"+str(depth)))
            depth += 1
	    
    log = Log(train_config)

 

    for id, config in train_config.items():
        if evaluate : ## WATCH OUT EVAL LOGGING WORKS ONLY WITH 1 SUPERVISED LAYER
            if config['mode'] == 'supervised':
                train_loader, test_loader = make_data_loaders(dataset_sup_config, config['batch_size'], device)
                criterion = nn.CrossEntropyLoss()
                test_loss, test_acc = evaluate_sup(model, criterion, test_loader, device)
                print(f'Accuracy of the network: {test_acc:.3f} %')
                print(f'Test loss: {test_loss:.3f}')
        else: 
            print("BLOCKS: ", config['blocks'])
            if config['mode'] == 'unsupervised':
                run_unsup(
                    config['nb_epoch'],
                    config['print_freq'],
                    config['batch_size'],
                    name_model,
                    dataset_unsup_config,
                    model,
                    device,
                    log.unsup[id],
                    blocks=config['blocks'],
                    save=save
                )
                print()
            elif config['mode'] == 'supervised':
                run_sup(
                    config['nb_epoch'],
                    config['print_freq'],
                    config['batch_size'],
                    config['lr'],
                    name_model,
                    dataset_sup_config,
                    model,
                    device,
                    log.sup[id],
                    blocks=config['blocks'],
                    save=save
                )
                #print()

            else:
                run_hybrid(
                    config['nb_epoch'],
                    config['print_freq'],
                    config['batch_size'],
                    config['lr'],
                    name_model,
                    dataset_sup_config,
                    model,
                    device,
                    log.sup[id],
                    blocks=config['blocks'],
                    save=save
                )

    save_logs(log, name_model)




if __name__ == '__main__':
    gc.collect()

    torch.cuda.empty_cache()
    params = parser.parse_args()
    name_model = params.preset if params.model_name is None else params.model_name
    blocks = load_presets(params.preset)
    dataset_sup_config = load_config_dataset(params.dataset_sup, params.validation)
    dataset_unsup_config = load_config_dataset(params.dataset_unsup, params.validation)
    if params.seed is not None:
        dataset_sup_config['seed'] = params.seed
        dataset_unsup_config['seed'] = params.seed

    if dataset_sup_config['seed'] is not None:
        seed_init_fn(dataset_sup_config['seed'])

    blocks = check_dimension(blocks, dataset_sup_config)

    print("BLOCKS: ", blocks)

    train_config = training_config(blocks, dataset_sup_config, dataset_unsup_config, params.training_mode,
                                   params.training_blocks)
    print("train_config: ", train_config)

    main(blocks, name_model, params.resume, params.save, params.evaluate, dataset_sup_config, dataset_unsup_config, train_config,
         params.gpu_id)
