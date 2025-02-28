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
import ast
import os
import sys
import uuid

import os.path as op
import json
from utils import load_presets, get_device, load_config_dataset, seed_init_fn, str2bool
from model import load_layers
from train import run_sup, run_unsup, check_dimension, training_config, run_hybrid
from log_m import Log, save_logs
import warnings
import copy

from utils import CustomStepLR, double_factorial
from engine_cl import evaluate_sup, train_sup, train_unsup, evaluate_unsup, getActivation, evaluate_sup_multihead
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
parser.add_argument('--folder-id', default=None,
                    type=str )
parser.add_argument('--parent-f-id', default=None,
                    type=str )

parser.add_argument('--dataset-unsup-1', choices=load_config_dataset(), default=None,
                    type=str, help='Dataset possibilities ' +
                                   ' | '.join(load_config_dataset()) +
                                   ' (default: None)')

parser.add_argument('--dataset-sup-1', choices=load_config_dataset(), default=None,
                    type=str, help='Dataset possibilities ' +
                                   ' | '.join(load_config_dataset()) +
                                   ' (default: None)')

parser.add_argument('--dataset-unsup-2', choices=load_config_dataset(), default=None,
                    type=str, help='Dataset possibilities ' +
                                   ' | '.join(load_config_dataset()) +
                                   ' (default: None)')

parser.add_argument('--dataset-sup-2', choices=load_config_dataset(), default=None,
                    type=str, help='Dataset possibilities ' +
                                   ' | '.join(load_config_dataset()) +
                                   ' (default: None)')

parser.add_argument('--training-mode', choices=['successive', 'consecutive', 'simultaneous'], default='consecuttive',   ###################
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
parser.add_argument('--topk-lock', default=False, type=str2bool, metavar='N',
                    help='')
parser.add_argument('--skip-1', default=False, type=str2bool, metavar='N',
                    help='Set to True if you want to skip the training on the first dataset and directly retrieve a model to train it again on the second dataset (you don \'t have to specify a preset if set True) ')
parser.add_argument('--classes-per-task', default=-1, type=int,
                    help='The continual learning is organized in tasks made up of different classes of the same dataset. Number of classes belonging to each task.')
parser.add_argument('--dataset-unsup', choices=load_config_dataset(),  default=None,
                    type=str, help='Dataset possibilities ' +
                                   ' | '.join(load_config_dataset()) +
                                   ' (default: None)')

parser.add_argument('--dataset-sup', choices=load_config_dataset(),  default=None,
                    type=str, help='Dataset possibilities ' +
                                   ' | '.join(load_config_dataset()) +
                                   ' (default: None)')

parser.add_argument('--head-sol', choices=[True, False], default='True',   ###################
                    type=str2bool, help='whether continual learning solution is on or off on linear layers' +
                                   ' | '.join(['on', 'off']) +
                                   ' (default: off)')

parser.add_argument('--cf-sol', default="True",   ###################
                    type=str2bool)
parser.add_argument('--top-k', default=0.8,   ###################
                    type=float)
parser.add_argument('--high-lr', default=0.2,   ###################
                    type=float)
parser.add_argument('--low-lr', default=0.8,   ###################
                    type=float)
parser.add_argument('--delta-w-interval', default=100,   ###################
                    type=float)
parser.add_argument('--t-criteria', default="mean",   ###################
                    type=str)
parser.add_argument('--heads-basis-t', default=0.6,   ###################
                    type=float)
parser.add_argument('--selected-classes', default="[[0,3],[5,7]]",   ###################
                    type=str)
parser.add_argument('--n-tasks', default=2,   ###################
                    type=int)
parser.add_argument('--evaluated-tasks', default="[0,1]",   ###################
                    type=str)
# we need first to pass both the datasets, the evaluation parameter is not needed, or it could be if we decide to validate just one model on one dataset. 
# after we passed both the datasets, train the model on the 1st dataset ( the resume all flag must be artificially set to false) and retrieved the model saved. The continual learning flag will cut the dataset, but it must be applied only 
# during the second training of the model. And so the evaluate must be set to true in the last iteration and continual learning again to false.

results = {"count": 0}


def main(blocks, name_model, resume, save, dataset_sup_config, dataset_unsup_config, train_config, gpu_id, evaluate, results, cl_hyper):
    device = get_device(gpu_id)
    model = load_layers(blocks, name_model, resume, dataset_sup_config=dataset_sup_config, batch_size=list(train_config.values())[-1]["batch_size"], cl_hyper=cl_hyper)

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
                if subsubl._get_name().__eq__("Linear"):
                    subsubl.register_forward_hook(getActivation("linear"+str(depth)))
            depth += 1
    
    
    
    log = Log(train_config)
    test_loss = 0
    test_acc = 0


    for id, config in train_config.items():
        print("CONFIG MODE: ", config['mode'])
        if evaluate:
            
            if config['mode'] == 'supervised' or config['mode'] == 'hybrid': ## WATCH OUT EVAL LOGGING WORKS ONLY WITH 1 SUPERVISED LAYER
                train_loader, test_loader = make_data_loaders(dataset_sup_config, config['batch_size'], device)
                criterion = nn.CrossEntropyLoss()
                if cl_hyper["head_sol"]:
                    res = evaluate_sup_multihead(model, criterion, test_loader, device)
                    test_loss = res[0]
                    test_acc = res[1]
                else: 
                    test_loss, test_acc= evaluate_sup(model, criterion, test_loader, device)
                print(f'Accuracy of the network on the 1st dataset: {test_acc:.3f} %')
                print(f'Test loss on the 1st dataset: {test_loss:.3f}')

                conv, R1 = model.convergence()
                if type(test_loss) ==  torch.Tensor:
                    metrics = {"test_loss":test_loss.item(), "test_acc": test_acc.item(), "convergence":conv, "R1":R1}
                else: 
                    metrics = {"test_loss":test_loss, "test_acc": test_acc, "convergence":conv, "R1":R1}
                metrics["dataset_sup"] = dataset_sup_config.copy()
                metrics["dataset_unsup"] = dataset_unsup_config.copy()

                                
                results["cl_hyper"] = cl_hyper
                results[f"eval_{results['count']%results['cl_hyper']['n_tasks']}"] = metrics.copy()
                results["count"] += 1
        else:
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
                    save=save, 
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
                    save=save,
                )
                result["dataset_sup"] = dataset_sup_config.copy()
                result["dataset_unsup"] = dataset_unsup_config.copy()
                result["train_config"] = train_config.copy()
                print("RESULT: ", result)
                results["R" + str(results["count"])] = result.copy()
                print(f"IN R" + str(results["count"]) + ": ", results)
                results["count"] += 1
            else:
                result = run_hybrid(
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
                    save=save, 
                )
                result["dataset_sup"] = dataset_sup_config.copy()
                result["dataset_unsup"] = dataset_unsup_config.copy()
                result["train_config"] = train_config.copy()
                print("RESULT: ", result)
                results["R" + str(results["count"])] = result.copy()
                print(f"IN R" + str(results["count"]) + ": ", results)
                results["count"] += 1
    results["model_config"] = blocks
    # save_logs(log, name_model)
    # print("Name Model: ", name_model)
    
    # datas = load_data(name_model, train_config)
    # for d in datas: 

    #     print("Datas: ", d)


def procedure(params, name_model, blocks, dataset_sup_config, dataset_unsup_config, evaluate, results):
    #print("type(params.cl_hyper): ", type(params.cl_hyper["selected_classes"]))
    if params.seed is not None:
        dataset_sup_config['seed'] = params.seed
        dataset_unsup_config['seed'] = params.seed

    if dataset_sup_config['seed'] is not None:
        seed_init_fn(dataset_sup_config['seed'])

    blocks = check_dimension(blocks, dataset_sup_config)

    train_config = training_config(blocks, dataset_sup_config, dataset_unsup_config, params.training_mode,
                                   params.training_blocks)
     
    main(blocks, name_model, params.resume, params.save, dataset_sup_config, dataset_unsup_config, train_config,
          params.gpu_id, evaluate, results, cl_hyper=params.cl_hyper)

def save_results(results, file):
    print("results: ", results)
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

def save_results_new(results, path, name):
    print("results: ", results)

    if not os.path.exists(path):
        print("MKDIR")
        os.mkdir(path)
    file = path + "/"+ name + ".json"
    print(file)
    with open(file, 'w') as f:
        json.dump(results, f, indent=4)

def random_n_classes(all_classes, n_classes):
    np.random.shuffle(all_classes)
    # select n classes indices to extract the classes
    classes = np.arange(0, n_classes)
    selected_classes = all_classes[classes]
    all_classes = np.delete(all_classes, classes)
    return all_classes, selected_classes

def task_training(params, name_model, blocks, selected_classes, dataset_sup, dataset_unsup, continual_learning, resume):
    #all_classes, selected_classes = random_n_classes(all_classes, n_classes)
                
    # selected_classes = selected_classes.tolist()
    # selected_classes = [2,8]
    
    print(selected_classes)
    dataset_sup["selected_classes"] = selected_classes
    dataset_unsup["selected_classes"] = selected_classes

    params.continual_learning = continual_learning
    params.resume = resume
    evaluate = False
    procedure(params, name_model, blocks, dataset_sup, dataset_unsup, evaluate, results)

if __name__ == '__main__':


    
    params = parser.parse_args()
    folder_id = params.folder_id
    name_model = params.preset if params.model_name is None else params.model_name
    name_model = name_model + str(uuid.uuid4())
    #name_model = "C100_2C_CLb50abfcf-7c09-4b6f-a581-dc7b529dd310"
    blocks = load_presets(params.preset)
    classes_per_task = params.classes_per_task
    resume = params.resume
   
    print(params.selected_classes)

    
    
    if classes_per_task != None and (params.dataset_sup_2 != None or params.dataset_sup_1 != None):
        print("\n\n ########### WARNING ############\n\n")
        print(" Invalid combination of parameters, provide either: [--classes, --dataset-sup, --dataset-unsup] or [--dataset-sup-1, --dataset-unsup-1, --dataset-sup-2, --dataset-unsup-2]\nThe continual learning is implemented per tasks where each task is made up of different classes \n of the same dataset, so only one dataset will be considered.")
        print("\n\n ################################\n\n")



    if classes_per_task != -1: 

        cl_hyper = {
                'training_mode': params.training_mode,
                'cf_sol': params.cf_sol,
                'head_sol': params.head_sol,
                'top_k': params.top_k,
                "topk_lock": params.topk_lock,
                'high_lr': params.high_lr,
                'low_lr':params.low_lr,
                't_criteria': params.t_criteria,
                'delta_w_interval': params.delta_w_interval,
                'heads_basis_t': params.heads_basis_t,
                "classes_per_task": params.classes_per_task,
                "n_tasks": params.n_tasks, 
                'selected_classes': eval(params.selected_classes),
                "evaluated_tasks": eval(params.evaluated_tasks), 
                
                

            }
        print(cl_hyper)
        params.training_mode = cl_hyper["training_mode"]
        params.cl_hyper = cl_hyper
        
        
        dataset_sup_ground  = load_config_dataset(params.dataset_sup, params.validation, params.continual_learning)
        dataset_unsup_ground = load_config_dataset(params.dataset_unsup, params.validation, params.continual_learning)
        
        
        out_channels = dataset_sup_ground["out_channels"]
        dataset_sup_ground["old_dataset_size"] = dataset_sup_ground["width"]
        dataset_unsup_ground["old_dataset_size"] = dataset_unsup_ground["width"]

        dataset_sup_ground["n_classes"] = classes_per_task
        dataset_unsup_ground["n_classes"] = classes_per_task

        dataset_sup_ground["out_channels"] = classes_per_task
        dataset_unsup_ground["out_channels"] = classes_per_task

        all_classes = np.arange(0, out_channels)
        
       
        dataset_sup_1 = dataset_sup_ground.copy()
        dataset_unsup_1 = dataset_unsup_ground.copy()

        if out_channels >=  2*classes_per_task:

            # TASK 1
            skip = params.skip_1
            print("task 1")
            if not skip: 
                selected_classes = cl_hyper["selected_classes"][0]
                task_training(params, name_model, blocks, selected_classes, dataset_sup_1, dataset_unsup_1, continual_learning=False, resume=False)

            else: 
                all_classes, selected_classes = random_n_classes(all_classes, classes_per_task)
                
                # selected_classes = selected_classes.tolist()
                # selected_classes = [2,8]

                selected_classes = cl_hyper["selected_classes"][0]
                print(selected_classes)

                dataset_sup_1["selected_classes"] = selected_classes
                dataset_unsup_1["selected_classes"] = selected_classes
                params.continual_learning = False
                evaluate = True
                procedure(params, name_model, blocks, dataset_sup_1, dataset_unsup_1, evaluate, results)
            
            for task_num in range(1, cl_hyper["n_tasks"]):
                print("################################## TASK " + str(task_num)+ " ############################################")

                selected_classes = cl_hyper["selected_classes"][task_num]
                dataset_sup_x = dataset_sup_ground.copy()
                dataset_unsup_x = dataset_unsup_ground.copy()

                task_training(params, name_model, blocks, selected_classes, dataset_sup_x, dataset_unsup_x, continual_learning=True, resume=resume)

           
            # EVALUATION PHASE
            params.continual_learning = False
            evaluate = True
            if max(cl_hyper["evaluated_tasks"]) >= cl_hyper['n_tasks']:
                cl_hyper["evaluated_tasks"] = list(range(cl_hyper['n_tasks']))
            for task_num in cl_hyper["evaluated_tasks"]:
                print("################################## TASK " + str(task_num)+ " ############################################")

                selected_classes = cl_hyper["selected_classes"][task_num]
                dataset_sup_x = dataset_sup_ground.copy()
                dataset_unsup_x = dataset_unsup_ground.copy()
                dataset_sup_x["selected_classes"] = selected_classes
                dataset_unsup_x["selected_classes"] = selected_classes

                procedure(params, name_model, blocks, dataset_sup_x, dataset_unsup_x, evaluate, results)

        
            save_results_new(results, f"{params.parent_f_id}/TASKS_CL_{params.dataset_sup.split('_')[0] +  folder_id}", name_model)
        else: 
            print("Error: Not enough available classes to be organized in tasks of classes_per_task")



    else:
        cl_hyper = {
                'training_mode': params.training_mode,
                'cf_sol': params.cf_sol,
                'head_sol': params.head_sol,
                'top_k': params.top_k,
                "topk_lock": params.topk_lock,
                'high_lr': params.high_lr,
                'low_lr':params.low_lr,
                't_criteria': params.t_criteria,
                'delta_w_interval': params.delta_w_interval,
                'heads_basis_t': params.heads_basis_t,
                "n_tasks": params.n_tasks, 
                

            }
        print(cl_hyper)
        params.training_mode = cl_hyper["training_mode"]
        params.cl_hyper = cl_hyper
        # DATASET 1


        resume = params.resume
        skip =  params.skip_1
        skip = False
        dataset_sup_1 = load_config_dataset(params.dataset_sup_1, params.validation, params.continual_learning)


        if not skip: 
            params.continual_learning = False
            params.resume = None
            dataset_sup_1 = load_config_dataset(params.dataset_sup_1, params.validation, params.continual_learning)
            dataset_unsup_1 = load_config_dataset(params.dataset_unsup_1, params.validation, params.continual_learning)
            procedure(params, name_model, blocks,dataset_sup_1, dataset_unsup_1, False, results)
        else: 
            params.continual_learning = False
            evaluate = True
            dataset_sup_1 = load_config_dataset(params.dataset_sup_1, params.validation, params.continual_learning)
            dataset_unsup_1 = load_config_dataset(params.dataset_unsup_1, params.validation, params.continual_learning)
            procedure(params, name_model, blocks, dataset_sup_1, dataset_unsup_1, evaluate, results)

        # DATASET 2

        params.continual_learning = True
        params.resume = resume
        evaluate = False

        dataset_sup_2 = load_config_dataset(params.dataset_sup_2, params.validation, params.continual_learning)
        dataset_unsup_2 = load_config_dataset(params.dataset_unsup_2, params.validation, params.continual_learning)
        dataset_sup_2["old_dataset_size"] = dataset_sup_1["width"]
        dataset_unsup_2["old_dataset_size"] = dataset_unsup_1["width"]
        
        procedure(params, name_model, blocks,dataset_sup_2, dataset_unsup_2, evaluate, results)

        # EVALUATION PHASE
       
        params.continual_learning = False
        evaluate = True
        procedure(params, name_model, blocks, dataset_sup_1, dataset_unsup_1, evaluate, results)
        
        results["model_name"] = name_model
        # file = "MULTD_CL.json"
        # save_results(results, file)
        save_results_new(results, f"{params.parent_f_id}/MULTD_CL_{params.dataset_sup_1.split('_')[0] + '_' + params.dataset_sup_2.split('_')[0]  + '_' + folder_id}", name_model)
