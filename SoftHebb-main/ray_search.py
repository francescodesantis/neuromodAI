
"""
The file contains the functions which allow for parameter parsing and configuration of the main attributes for the training. 
Additionally  it contains the main function which starts the execution. 
"""


import argparse
import copy
import os
import pdb

from utils import SEARCH, load_presets, get_device, load_config_dataset, merge_parameter, seed_init_fn, str2bool
from model import load_layers
import torch
from train import run_sup, run_unsup, check_dimension, training_config, run_hybrid
from log_m import Log
import ray
from ray import tune
#from ray.tune.suggest.basic_variant import BasicVariantGenerator
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune import CLIReporter
from functools import partial
import warnings
import numpy as np
from ray.tune.tuner import Tuner
from ray.tune.schedulers import ASHAScheduler
warnings.filterwarnings("ignore")

metric_names = ['train_loss', 'train_acc', 'test_loss', 'test_acc', 'convergence', 'R1']

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
                                   ' (default: consecutive)')

parser.add_argument('--resume', choices=[None, "all", "without_classifier"], default=None,
                    type=str, help='Resume Model ' +
                                   ' | '.join(["all", "without_classifier"]) +
                                   ' (default: None)')

parser.add_argument('--metric', choices=metric_names, default='test_acc',
                    type=str, help='Primary Metric' +
                                   ' | '.join(metric_names) +
                                   ' (default: test_acc)')

parser.add_argument('--training-blocks', default=None, nargs='+', type=int,
                    help='Selection of the blocks that will be trained')

parser.add_argument('--folder-name', default=None, type=str,
                    help='Name of the experiment')

parser.add_argument('--num-samples', default=1, type=int,
                    help='number of search into the hparams space')

parser.add_argument('--model-name', default=None, type=str, help='Model Name')

parser.add_argument('--validation-sup', default=False, type=str2bool, metavar='N',
                    help='')

parser.add_argument('--validation-unsup', default=False, type=str2bool, metavar='N',
                    help='')
parser.add_argument('--config', default='seed', type=str, metavar='N',
                    help='')

parser.add_argument('--gpu-exp', default=1, type=int, metavar='N',
                    help='')

parser.add_argument('--save-model', default=False, action='store_true',
                    help='Save model checkpoints, configs, etc')

parser.add_argument('--debug', default=False, action='store_true', help='Debug mode (ray local)')
parser.add_argument('--test', default=False, type=str2bool, metavar='N',
                    help='Set to True if you want to reduce the number of seed for ray search to just one to speed up training ')

default={}
print(parser)

#ok


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
        if params.test: 
            config = {
                'dataset_unsup': {
                    #'seed': tune.grid_search([0, 1, 2, 3])
                    'seed': tune.grid_search([0]) ###############################################
                }
            }
        else: 
            config = {
                'dataset_unsup': {
                    'seed': tune.grid_search([0, 1, 2, 3])
                    #'seed': tune.grid_search([0]) ###############################################
                }
            }
    print("config_name", config_name)
    print("config", config)
    return config

#
#
""" 
 The input parmeters are: 

  - params: the different kind of params are the ones specified above where the parsing happens. 
  - dataset_sup_config: specified in the --dataset-sup
  - dataset_unsup_config: specified in the --dataset-unsup
  - blocks: specified in the --preset option
  - config: --config contains the type of seed.

 """
def main(params, dataset_sup_config, dataset_unsup_config, blocks, config):
    for block_id, block in blocks.items():
        if block_id in config:
            blocks[block_id] = merge_parameter(block.copy(), config[block_id])
    #print("blocks", blocks)

    if "dataset_unsup" in config:
        dataset_unsup_config = merge_parameter(dataset_unsup_config, config['dataset_unsup'])

    if "dataset_sup" in config:
        dataset_sup_config = merge_parameter(dataset_sup_config, config['dataset_sup'])

    if dataset_unsup_config['seed'] is not None:
        seed_init_fn(dataset_unsup_config['seed'])

    device = get_device()

    blocks = check_dimension(blocks, dataset_sup_config)

    print("dataset_sup_config, dataset_unsup_config", dataset_sup_config, dataset_unsup_config)
    train_config = training_config(blocks, dataset_sup_config, dataset_unsup_config, params.training_mode,
                                   params.training_blocks)

    print("train_config", train_config)

    model = load_layers(blocks, params.name_model, params.resume)

    model.reset()

    model = model.to(device)

    log = Log(train_config)
    print("TRAIN_CONFIG: ", train_config)
    for id, config in train_config.items():
        if config['mode'] == 'unsupervised':
            run_unsup(
                config['nb_epoch'],
                config['print_freq'],
                config['batch_size'],
                params.name_model,
                dataset_unsup_config,
                model,
                device,
                log.unsup[id],
                blocks=config['blocks'],
                report=ray.train.report,
                save=params.save_model,
                reset=False,
                model_dir=ray.train.get_context().get_trial_dir(),
                #model_dir = ray.train.get_context()
            )
        elif config['mode'] == 'supervised':
            print('Running supervised')
            run_sup(
                config['nb_epoch'],
                config['print_freq'],
                config['batch_size'],
                config['lr'],
                params.name_model,
                dataset_sup_config,
                model,
                device,
                log.sup[id],
                blocks=config['blocks'],
                report=ray.train.report,
                save=params.save_model,
                model_dir=ray.train.get_context().get_trial_dir(),
                #model_dir = ray.train.get_context() 
            )
        else:
            run_hybrid(
                config['nb_epoch'],
                config['print_freq'],
                config['batch_size'],
                config['lr'],
                params.name_model,
                dataset_sup_config,
                model,
                device,
                log.sup[id],
                blocks=config['blocks'],
                report=ray.train.report,
                save=params.save_model,
                model_dir=ray.train.get_context().get_trial_dir(),
            )


if __name__ == '__main__':
    params = parser.parse_args()
    #ray.init(include_dashboard=False, local_mode=True)
    #print("Params:    " + params)
    #ray.init(_temp_dir="$HOME")
    config = get_config(params.config)

    params.name_model = params.preset if params.model_name is None else params.model_name  # TODO change this for better model storage
    blocks = load_presets(params.preset)

    dataset_sup_config = load_config_dataset(params.dataset_sup, params.validation_sup)
    dataset_unsup_config = load_config_dataset(params.dataset_unsup, params.validation_unsup)

    #if params.debug is True:
        #local_mode=True for debugging . It seems there's no need to init ray for these usecase
        #ray.init(local_mode=True)

    reporter = CLIReporter(max_progress_rows=12)
    for metric in metric_names:
        reporter.add_metric_column(metric)

    algo_search = BasicVariantGenerator()

    trial_exp = partial(
        main, params, dataset_sup_config, dataset_unsup_config, blocks
    )

    # TODO: use ray for model storing, as it is better aware of the different variants
    print(torch.cuda.device_count())
    #print(trial_exp)
    #print(params.metric)
    #print(algo_search)
    #print(config)
    #print(reporter)
    #print(params.num_samples)
    #print(params.folder_name)

    scheduler = ASHAScheduler(
    grace_period=20, reduction_factor=3, max_t=100_000)

    
    # tuner = tune.Tuner(
    #     trial_exp,
    #     # tune.with_resources(trial_exp,
    #     #     resources={"cpu": 2, "gpu": torch.cuda.device_count() }
    #     # ),
    #     param_space=config,
    #     tune_config=tune.TuneConfig(
    #         metric=params.metric,
    #         mode='min' if params.metric.endswith('loss') else 'max',
    #         scheduler=scheduler,
    #         num_samples=params.num_samples,
    #         search_alg=algo_search,        
    #         #local_dir=SEARCH,
    #         #name=params.folder_name),
    #     ),
        
    #     #run_config=tune.TuneConfig(progress_reporter=reporter),
    #     )

    #t = Tuner(tuner)
    # results = tuner.fit()

    analysis = tune.run(
        trial_exp,
        resources_per_trial={
            "cpu": os.cpu_count(),
            "gpu": torch.cuda.device_count()
        },
        metric=params.metric,
        mode='min' if params.metric.endswith('loss') else 'max',
        search_alg=algo_search,
        config=config,
        progress_reporter=reporter,
        num_samples=params.num_samples,
        storage_path=SEARCH,
        name=params.folder_name)
        
    # analysis = tune.Tuner(
    #     tune.with_resources(trial_exp, {
    #         "cpu": os.cpu_count(),
    #         "gpu": torch.cuda.device_count()
    #     }),
    #     tune_config=tune.TuneConfig(
        
    #     metric=params.metric,
    #     mode='min' if params.metric.endswith('loss') else 'max',
    #     search_alg=algo_search,
    #     num_samples=params.num_samples,
        
    #     progress_reporter=reporter,
    #     storage_path=SEARCH,
    #     name=params.folder_name
    #     ), 
    #     param_space=config,
    # )
    # results = analysis.fit()
    #
       