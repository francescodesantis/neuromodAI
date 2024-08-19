#This file contains the implementation of the continual learnign methods necessary

#First thing we need to be able to train a model, save it and retrieve it to train it and test it on another 
#dataset. To train it we can simply use the multi_layer.py file, here we retrieve the model and train it on another dataset.
#The training for the unsupervised part is performed by setting the number of epochs to 1 directly in the learning
#rate scheduler. For the supervised part it is set to nb_epoch value which is specified in the preset.json. 
#The files are going to be saved inside the Training folder with path 
# neuromodAI/SoftHebb-main/Training/results/hebb/result/network/2SoftHebbCnnCIFAR/models/checkpoint.pth.tar 
# we need to add a command which allows us to specify the different datasets and also a flag which tells us if we are performing continual leanring or not.
# at this point we have to see decide the size of the training images: we could start with a dataset which is bigger and the second one smaller
# so we need to either upscale it or downscale it. 

import argparse

from utils import load_presets, get_device, load_config_dataset, seed_init_fn, str2bool
from model import load_layers
from train import run_sup, run_unsup, check_dimension, training_config, run_hybrid
from log import Log, save_logs
import warnings

from utils import CustomStepLR, double_factorial
from model import save_layers, HebbianOptimizer, AggregateOptim
from engine import train_sup, train_unsup, evaluate_unsup, evaluate_sup
from dataset import make_data_loaders
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Multi layer Hebbian Training Continual Learning  implementation')

parser.add_argument('--continual_learning', choices=['True', 'False'], default='False',
                    type=str)


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

parser.add_argument('--dataset-unsup-2', choices=load_config_dataset(), default='MNIST',
                    type=str, help='Dataset possibilities ' +
                                   ' | '.join(load_config_dataset()) +
                                   ' (default: MNIST)')

parser.add_argument('--dataset-sup-2', choices=load_config_dataset(), default='MNIST',
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




def main(blocks, name_model, resume, save, dataset_sup_config, dataset_unsup_config, train_config, gpu_id, evaluate):
    device = get_device(gpu_id)
    model = load_layers(blocks, name_model, resume)
    
    model = model.to(device)
    print("FIRST LAYER INFO: ", model)
    log = Log(train_config)

    for id, config in train_config.items():
        if evaluate:
            evaluate_sup_CL(
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
                save=save )
        elif config['mode'] == 'unsupervised':
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
    params = parser.parse_args()
    name_model = params.preset if params.model_name is None else params.model_name
    if params.continual_learning == "True":
        #we need to load the model specified in model_name, see what is the image size accepted and 
        # then resize the whole new dataset
        print("")
    blocks = load_presets(params.preset)
    dataset_sup_config = load_config_dataset(params.dataset_sup, params.validation)
    dataset_unsup_config = load_config_dataset(params.dataset_unsup, params.validation)
    if params.seed is not None:
        dataset_sup_config['seed'] = params.seed
        dataset_unsup_config['seed'] = params.seed

    if dataset_sup_config['seed'] is not None:
        seed_init_fn(dataset_sup_config['seed'])

    blocks = check_dimension(blocks, dataset_sup_config)

    train_config = training_config(blocks, dataset_sup_config, dataset_unsup_config, params.training_mode,
                                   params.training_blocks)

    main(blocks, name_model, params.resume, params.save, dataset_sup_config, dataset_unsup_config, train_config,
         params.gpu_id, params.evaluate)


def evaluate_sup_CL(
        final_epoch: int,
        print_freq: int,
        batch_size: int,
        lr: float,
        folder_name: str,
        dataset_config: dict,
        model,
        device,
        log,
        blocks,
        learning_mode: str = 'BP',
        save_batch: bool = False,
        save: bool = True,
        report=None,
        plot_fc=None,
        model_dir=None
):
    """
    Supervised training of BP blocks of one model

    """

    print('\n', '********** Supervised learning of blocks %s **********' % blocks)
    print("SAVING FOLDER FOR SUP: ", folder_name)

    train_loader, test_loader = make_data_loaders(dataset_config, batch_size, device)

    criterion = nn.CrossEntropyLoss()
    log_batch = log.new_log_batch()
    if all([model.get_block(b).is_hebbian() for b in blocks]):
        # optimizer, scheduler, log_batch = None, None, None
        optimizer, scheduler = None, None
    else:
        # criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)  # , weight_decay=1e-6)
        scheduler = CustomStepLR(optimizer, final_epoch)

    for epoch in range(1, final_epoch + 1):
        #measures, lr = train_sup(model, criterion, optimizer, train_loader, device, log_batch, learning_mode, blocks)

        if scheduler is not None:
            scheduler.step()

        if epoch % print_freq == 0 or epoch == final_epoch or epoch == 1:

            loss_test, acc_test = evaluate_sup(model, criterion, test_loader, device)

            log_batch = log.step(epoch, log_batch, loss_test, acc_test, lr, save_batch)

            if report is not None:
                _, train_loss, train_acc, test_loss, test_acc = log.data[-1]
                conv, R1 = model.convergence()
                metrics = {"train_loss":train_loss, "train_acc":train_acc, "test_loss":test_loss, "test_acc": test_acc, "convergence":conv, "R1":R1}
                report(metrics)
                
            else:
                log.verbose()

            if save:
                save_layers(model, folder_name, epoch, blocks, storing_path=model_dir)

            if plot_fc is not None:
                for block in blocks:
                    plot_fc(model, block)
