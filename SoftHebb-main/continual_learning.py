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
from utils import load_presets, get_device, load_config_dataset, seed_init_fn, str2bool
from model import load_layers
from train import run_sup, run_unsup, check_dimension, training_config, run_hybrid
from log import Log, save_logs
import warnings
import copy

from utils import CustomStepLR, double_factorial
from model import save_layers, HebbianOptimizer, AggregateOptim
from engine import train_sup, train_unsup, evaluate_unsup, evaluate_sup
from dataset import make_data_loaders
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from nb_utils import load_data


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Multi layer Hebbian Training Continual Learning  implementation')

parser.add_argument('--continual_learning', choices=[True, False], default=False,
                    type=str2bool)


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
                    help='')

parser.add_argument('--gpu-id', default=0, type=int, metavar='N',
                    help='Id of gpu selected for training (default: 0)')

parser.add_argument('--save', default=True, type=str2bool, metavar='N',
                    help='')

parser.add_argument('--validation', default=False, type=str2bool, metavar='N',
                    help='')

parser.add_argument('--evaluate', default=False, type=str2bool, metavar='N',
                    help='')
parser.add_argument('--skip-1', default=False, type=str2bool, metavar='N',
                    help='Set to True if you want to skip the training on the first dataset and directly retrieve a model to train it again on the second dataset (you don \'t have to specify a preset if set True) ')
parser.add_argument('--classes', default=None, type=int,
                    help='The continual learning is organized in tasks made up of different classes of the same dataset. Number of classes belonging to each task.')
parser.add_argument('--dataset-unsup', choices=load_config_dataset(),  default=None,
                    type=str, help='Dataset possibilities ' +
                                   ' | '.join(load_config_dataset()) +
                                   ' (default: MNIST)')

parser.add_argument('--dataset-sup', choices=load_config_dataset(),  default=None,
                    type=str, help='Dataset possibilities ' +
                                   ' | '.join(load_config_dataset()) +
                                   ' (default: MNIST)')
# we need first to pass both the datasets, the evaluation parameter is not needed, or it could be if we decide to validate just one model on one dataset. 
# after we passed both the datasets, train the model on the 1st dataset ( the resume all flag must be artificially set to false) and retrieved the model saved. The continual learning flag will cut the dataset, but it must be applied only 
# during the second training of the model. And so the evaluate must be set to true in the last iteration and continual learning again to false.



def main(blocks, name_model, resume, save, dataset_sup_config, dataset_unsup_config, train_config, gpu_id, evaluate, results):
    device = get_device(gpu_id)
    model = load_layers(blocks, name_model, resume)
        
    #model = copy.deepcopy(model_og)
    
    model = model.to(device)
    log = Log(train_config)
    test_loss = 0
    test_acc = 0
    for id, config in train_config.items():
        if evaluate:
            train_loader, test_loader = make_data_loaders(dataset_sup_config, config['batch_size'], device)
            criterion = nn.CrossEntropyLoss()
            test_loss, test_acc = evaluate_sup(model, criterion, test_loader, device)
            print(f'Accuracy of the network on the 1st dataset: {test_acc:.3f} %')
            print(f'Test loss on the 1st dataset: {test_loss:.3f}')

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

def procedure(params, name_model, blocks, dataset_sup_config, dataset_unsup_config, evaluate, results):

    if params.seed is not None:
        dataset_sup_config['seed'] = params.seed
        dataset_unsup_config['seed'] = params.seed

    if dataset_sup_config['seed'] is not None:
        seed_init_fn(dataset_sup_config['seed'])

    blocks = check_dimension(blocks, dataset_sup_config)

    train_config = training_config(blocks, dataset_sup_config, dataset_unsup_config, params.training_mode,
                                   params.training_blocks)

    main(blocks, name_model, params.resume, params.save, dataset_sup_config, dataset_unsup_config, train_config,
         params.gpu_id, evaluate, results)

def save_results(results, file):
    with open(file, 'a+') as f:
        try:
            f.seek(0)
            old = json.load(f)
        except json.JSONDecodeError:
            old = {}
    with open(file, 'r+') as f:
        try:
            old = json.load(f)
        except json.JSONDecodeError:
            old = {}
        
        if old.get("T1") is None:
            old["T1"] = results
        else: 
            last_key = list(old.keys())[-1]
            new_key = "T" + str(int(last_key[1:]) + 1)
            old[new_key] = results

        f.seek(0)
        f.truncate() 

        json.dump(old, f, indent=4)

def random_n_classes(all_classes, n_classes):
    np.random.shuffle(all_classes)
    # select n classes indices to extract the classes
    classes = np.arange(0, n_classes)
    selected_classes = all_classes[classes]
    all_classes = np.delete(all_classes, classes)
    return all_classes, selected_classes

if __name__ == '__main__':



    params = parser.parse_args()
    name_model = params.preset if params.model_name is None else params.model_name
    blocks = load_presets(params.preset)
    n_classes = params.classes
    resume = params.resume
    new_model = False
    results = {}


    if n_classes != None and (params.dataset_sup_2 != None or params.dataset_sup_1 != None):
        print("\n\n ########### WARNING ############\n\n")
        print(" Invalid combination of parameters, provide either: [--classes, --dataset-sup, --dataset-unsup] or [--dataset-sup-1, --dataset-unsup-1, --dataset-sup-2, --dataset-unsup-2]\nThe continual learning is implemented per tasks where each task is made up of different classes \n of the same dataset, so only one dataset will be considered.")
        print("\n\n ################################\n\n")


    if n_classes != None: 
        
        dataset_sup_config = load_config_dataset(params.dataset_sup, params.validation, params.continual_learning)
        dataset_unsup_config = load_config_dataset(params.dataset_unsup, params.validation, params.continual_learning)
        out_channels = dataset_sup_config["out_channels"]
        dataset_sup_config["old_dataset_size"] = dataset_sup_config["width"]
        dataset_unsup_config["old_dataset_size"] = dataset_unsup_config["width"]

        dataset_sup_config["n_classes"] = n_classes
        dataset_unsup_config["n_classes"] = n_classes

        dataset_sup_config["out_channels"] = n_classes
        dataset_unsup_config["out_channels"] = n_classes

        all_classes = np.arange(0, out_channels)

        if out_channels >=  2*n_classes:

            # TASK 1
            skip = params.skip_1

            if not skip: 
                all_classes, selected_classes = random_n_classes(all_classes, n_classes)
                selected_classes = [2, 1]
                dataset_sup_config["selected_classes"] = selected_classes
                dataset_unsup_config["selected_classes"] = selected_classes

                params.continual_learning = False
                params.resume = None
                evaluate = False
                procedure(params, name_model, blocks, dataset_sup_config, dataset_unsup_config, evaluate, results)

            # TASK 2
            all_classes, selected_classes = random_n_classes(all_classes, n_classes)
            selected_classes = [2, 1]

            dataset_sup_config["selected_classes"] = selected_classes
            dataset_unsup_config["selected_classes"] = selected_classes

            params.continual_learning = True
            params.resume = resume
            evaluate = False
            new_model = True
            name_model = name_model + "_CLM"
            procedure(params, name_model, blocks, dataset_sup_config, dataset_unsup_config, evaluate, results)

            # EVALUATION PHASE
            params.continual_learning = False
            evaluate = True
            new_model = True
            procedure(params, name_model, blocks, dataset_sup_config, dataset_unsup_config, evaluate, results)

            file = "TASKS_CL.json"
            save_results(results, file)
        else: 
            print("Error: Not enough available classes to be organized in tasks of n_classes")



    else:
        # DATASET 1

        dataset_sup_config_1 = load_config_dataset(params.dataset_sup, params.validation, params.continual_learning)
        dataset_unsup_config_1 = load_config_dataset(params.dataset_unsup, params.validation, params.continual_learning)
        resume = params.resume
        skip = params.skip_1

        if not skip: 
            params.continual_learning = False
            params.resume = None
            procedure(params, name_model, blocks,dataset_sup_config_1, dataset_unsup_config_1, False, results)

        # DATASET 2

        dataset_sup_config_2 = load_config_dataset(params.dataset_sup_2, params.validation, params.continual_learning)
        dataset_unsup_config_2 = load_config_dataset(params.dataset_unsup_2, params.validation, params.continual_learning)
        dataset_sup_config_2["old_dataset_size"] = dataset_sup_config_1["width"]
        dataset_unsup_config_2["old_dataset_size"] = dataset_unsup_config_1["width"]

        params.continual_learning = True
        params.resume = resume
        evaluate = False
        name_model = name_model + "_CLM"
        procedure(params, name_model, blocks,dataset_sup_config_2, dataset_unsup_config_2, evaluate, results)

        # EVALUATION PHASE
       
        params.continual_learning = False
        evaluate = True
        procedure(params, name_model, blocks, dataset_sup_config_1, dataset_unsup_config_1, evaluate, results)
        
        file = "MULTD_CL.json"
        save_results(results, file)
