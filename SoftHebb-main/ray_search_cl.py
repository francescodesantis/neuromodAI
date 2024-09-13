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


# In order to train the best models on a new dataset and then evaluate them on the first one I have to skip the first training
# and go directly to the second one, how should I do so? With a flag --skip-1.
# Ok so when doing continual learning with the best models we need to set the skip-1 flag to true. 
# 

import argparse
import os.path as op
import json
from utils import SEARCH, merge_parameter, load_presets, get_device, load_config_dataset, seed_init_fn, str2bool
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
from nb_utils import load_data
import ray
from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune import CLIReporter
from ray.tune.tuner import Tuner
from ray.tune.schedulers import ASHAScheduler
from functools import partial

metric_names = ['train_loss', 'train_acc', 'test_loss', 'test_acc', 'convergence', 'R1']


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Multi layer Hebbian Training Continual Learning  implementation')

parser.add_argument('--continual_learning', choices=[True, False], default=False,
                    type=str2bool)

parser.add_argument('--metric', choices=metric_names, default='test_acc',
                    type=str, help='Primary Metric' +
                                   ' | '.join(metric_names) +
                                   ' (default: test_acc)')
parser.add_argument('--config', default='seed', type=str, metavar='N',
                    help='')

parser.add_argument('--preset', choices=load_presets(), default=None,
                    type=str, help='Preset of hyper-parameters ' +
                                   ' | '.join(load_presets()) +
                                   ' (default: None)')

parser.add_argument('--dataset-unsup-1', choices=load_config_dataset(), default='MNIST',
                    type=str, help='Dataset possibilities ' +
                                   ' | '.join(load_config_dataset()) +
                                   ' (default: MNIST)')

parser.add_argument('--dataset-sup-1', choices=load_config_dataset(), default='MNIST',
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
parser.add_argument('--num-samples', default=1, type=int,
                    help='number of search into the hparams space')

parser.add_argument('--gpu-id', default=0, type=int, metavar='N',
                    help='Id of gpu selected for training (default: 0)')
parser.add_argument('--gpu-exp', default=1, type=int, metavar='N',
                    help='')
parser.add_argument('--save', default=True, type=str2bool, metavar='N',
                    help='')

parser.add_argument('--validation', default=False, type=str2bool, metavar='N',
                    help='')
parser.add_argument('--validation-sup', default=False, type=str2bool, metavar='N',
                    help='')
parser.add_argument('--evaluate', default=False, type=str2bool, metavar='N',
                    help='')
parser.add_argument('--skip-1', default=False, type=str2bool, metavar='N',
                    help='Set to True if you want to skip the training on the first dataset and directly retrieve a model to train it again on the second dataset (you don \'t have to specify a preset if set True) ')

# we need first to pass both the datasets, the evaluation parameter is not needed, or it could be if we decide to validate just one model on one dataset. 
# after we passed both the datasets, train the model on the 1st dataset ( the resume all flag must be artificially set to false) and retrieved the model saved. The continual learning flag will cut the dataset, but it must be applied only 
# during the second training of the model. And so the evaluate must be set to true in the last iteration and continual learning again to false.


def get_config(config_name):
    
    if config_name == 'regimes':
        t_invert_search = [1.25 ** (x - 50) for x in range(100)]
        softness_search = ["soft", "softkrotov"]
        seeds = [0, 1, 2]
        configs = []
        for i_softness in softness_search:
            for i_t_invert in t_invert_search:
                for i_seed in seeds:
                    i_config = {
                        f'b{i_layer}': {
                            "layer": {
                                't_invert': i_t_invert,
                                "softness": i_softness,
                            }
                        } for i_layer in range(3)}
                    i_config['dataset_unsup'] = {
                        'seed': i_seed,
                    }
                    configs.append(i_config)

        config = tune.grid_search(configs)

    elif config_name == 'radius':
        config = {
            'b0': {
                "layer": {
                    'radius': tune.grid_search([1.25 ** (x - 10) for x in range(27)]),
                }
            },
            'dataset_unsup': {
                'seed': tune.grid_search([0, 1, 2]),
            }
        }
    elif config_name == 'one_seed':
        config = {
            'dataset_unsup': {
                'seed': 0
            }
        }
    else:
        config = {
            'dataset_unsup': {
                #'seed': tune.grid_search([0, 1, 2, 3])
                'seed': tune.grid_search([0 ]) ###############################################
            }
        }
    print("config_name", config_name)
    print("config", config)
    return config


def main(blocks, name_model, resume, save, dataset_sup_config, dataset_unsup_config, train_config, gpu_id, evaluate, results, config):
    device = get_device(gpu_id)
    model = load_layers(blocks, name_model, resume)
    ##################################################
    model.reset()
    if "dataset_unsup" in config:
        dataset_unsup_config = merge_parameter(dataset_unsup_config, config['dataset_unsup'])

    if "dataset_sup" in config:
        dataset_sup_config = merge_parameter(dataset_sup_config, config['dataset_sup'])

    if dataset_unsup_config['seed'] is not None:
        seed_init_fn(dataset_unsup_config['seed'])

    #################################################
    model = model.to(device)
    log = Log(train_config)
    loss = 0
    accuracy = 0
    for id, config in train_config.items():
        if evaluate:
            train_loader, test_loader = make_data_loaders(dataset_sup_config, config['batch_size'], device)
            criterion = nn.CrossEntropyLoss()
            test_loss, test_acc = evaluate_sup(model, criterion, test_loader, device)
            print(f'Accuracy of the network on the 1st dataset: {accuracy:.3f} %')
            print(f'Test loss on the 1st dataset: {loss:.3f}')

            conv, R1 = model.convergence()
            if type(test_loss) ==  torch.Tensor:
                metrics = {"test_loss":test_loss.item(), "test_acc": test_acc.item(), "convergence":conv, "R1":R1}
            else: 
                metrics = {"test_loss":test_loss, "test_acc": test_acc, "convergence":conv, "R1":R1}
            metrics["dataset_sup"] = dataset_sup_config
            metrics["dataset_unsup"] = dataset_unsup_config
            results["eval"] = metrics
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
            result = run_sup(
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
            result["dataset_unsup"] = dataset_unsup_config
            result["train_config"] = train_config
            if results.get("R1") == None: 
                results["R1"] = result
            else: 
                results["R2"] = result
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
    print("Name Model: ", name_model)
    
    datas = load_data(name_model, train_config)
    for d in datas: 

        print("Datas: ", d)

def procedure(params, blocks, dataset_sup_config, dataset_unsup_config, evaluate, results, config):

    if params.seed is not None:
        dataset_sup_config['seed'] = params.seed
        dataset_unsup_config['seed'] = params.seed

    if dataset_sup_config['seed'] is not None:
        seed_init_fn(dataset_sup_config['seed'])

    blocks = check_dimension(blocks, dataset_sup_config)

    train_config = training_config(blocks, dataset_sup_config, dataset_unsup_config, params.training_mode,
                                   params.training_blocks)

    main(blocks, name_model, params.resume, params.save, dataset_sup_config, dataset_unsup_config, train_config,
         params.gpu_id, evaluate, results, config)



if __name__ == '__main__':

    params = parser.parse_args()
    name_model = params.preset if params.model_name is None else params.model_name
    blocks = load_presets(params.preset)

    params.continual_learning = False
    dataset_sup_config_1 = load_config_dataset(params.dataset_sup_1, params.validation, params.continual_learning)
    dataset_unsup_config_1 = load_config_dataset(params.dataset_unsup_1, params.validation, params.continual_learning)

    params.continual_learning = True
    dataset_sup_config_2 = load_config_dataset(params.dataset_sup_2, params.validation, params.continual_learning)
    dataset_unsup_config_2 = load_config_dataset(params.dataset_unsup_2, params.validation, params.continual_learning)

    results = {}
    dataset_sup_config_2["old_dataset_size"] = dataset_sup_config_1["width"]
    dataset_unsup_config_2["old_dataset_size"] = dataset_unsup_config_1["width"]

    resume = params.resume

    config = get_config(params.config)
    reporter = CLIReporter(max_progress_rows=12)
    for metric in metric_names:
        reporter.add_metric_column(metric)

    algo_search = BasicVariantGenerator()

    scheduler = ASHAScheduler(
    grace_period=20, reduction_factor=3, max_t=100_000)

    # DATASET 1

    params.continual_learning = False
    params.resume = None
    #procedure(params, blocks,dataset_sup_config_1, dataset_unsup_config_1, False, results)
    trial_exp_1 = partial(
        procedure, params, blocks, dataset_sup_config_1, dataset_unsup_config_1, False, results
    )

    analysis = tune.run(
        trial_exp_1,
        resources_per_trial={
            "cpu": 4,
            "gpu": torch.cuda.device_count()
        },
        metric=params.metric,
        mode='min' if params.metric.endswith('loss') else 'max',
        search_alg=algo_search,
        config=config,
        progress_reporter=reporter,
        num_samples=params.num_samples,
        local_dir=SEARCH,
        name=params.model_name)
    

    # DATASET 2


    params.continual_learning = True
    params.resume = resume
    #procedure(params, blocks,dataset_sup_config_2, dataset_unsup_config_2, False, results)
    trial_exp_2 = partial(
        procedure, params, dataset_sup_config_2, dataset_unsup_config_2, False, blocks, results
    )

    config = get_config(params.config)
    reporter = CLIReporter(max_progress_rows=12)
    for metric in metric_names:
        reporter.add_metric_column(metric)

    algo_search = BasicVariantGenerator()

    scheduler = ASHAScheduler(
    grace_period=20, reduction_factor=3, max_t=100_000)

    analysis = tune.run(
        trial_exp_2,
        resources_per_trial={
            "cpu": 4,
            "gpu": torch.cuda.device_count()
        },
        metric=params.metric,
        mode='min' if params.metric.endswith('loss') else 'max',
        search_alg=algo_search,
        config=config,
        progress_reporter=reporter,
        num_samples=params.num_samples,
        local_dir=SEARCH,
        name=params.model_name)

    # EVALUATION PHASE
    params.continual_learning = False
    procedure(params, blocks,dataset_sup_config_1, dataset_unsup_config_1, True, results)
    

    

    print("RESULTS: ", results)
    data_candidate = "Continual_learning"
    DATA = op.realpath(op.expanduser(data_candidate))

    with open('CL_RES.json', 'a+') as f:
        try:
            f.seek(0)
            old = json.load(f)
        except json.JSONDecodeError:
        # If the file is empty or not valid JSON, start with an empty dictionary
            old = {}
    with open('CL_RES.json', 'r+') as f:
        try:
        # Load the existing data
            old = json.load(f)
        except json.JSONDecodeError:
            old = {}
        
        if old.get("T1") is None:
            old["T1"] = results
        else: 
            last_key = list(old.keys())[-1]
            new_key = "T" + str(int(last_key[1:]) + 1)
            old[new_key] = results
        #old.update(new_data)

        f.seek(0)
        f.truncate() 

        json.dump(old, f, indent=4)


    


