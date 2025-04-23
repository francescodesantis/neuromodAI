import ast
import itertools
import json
import os
import shlex
import subprocess
import time
import random
import uuid
import numpy as np
from t_hyper import classes_per_task, n_experiments, n_tasks, dataset, evaluated_tasks, folder_id, data_num, dataset2, cl_hyper, TEST, parent_f_id, USER
def folder_check(path):
    print(os.path.exists(f"/leonardo_work/{USER}/rcasciot/neuromodAI/SoftHebb-main/" + path))
    print(f"/leonardo_work/{USER}/rcasciot/neuromodAI/SoftHebb-main/" + path)
    return os.path.isdir("../SoftHebb-main/" + path)
def execute_bash_command(evaluated_tasks: list, n_tasks: int, command: str, classes=[]):
    modes = ["successive", "consecutive", "simultaneous"]
    lrs = [(0.0, 1.0), (2000, 1.0), (0.2, 0.8)]
    sols = [(True, True), (False, True), (False, False),  (True, False)]
    topks = [0.1, 0.2, 0.5, 0.7, 0.85, 0.9, 1.0]
    delta_w_intervals = [20, 100, 300]
    lr = lrs[2]
    mode = modes[1]
   
    
    for sol in sols:
        cl_hyper['cf_sol'] = sol[0]
        cl_hyper['head_sol'] = sol[1]
        cl_hyper['classes_per_task'] = classes_per_task
        
        if data_num == 1: 
            for sc in classes: # this corresponds to how many experiments we want to run
            
                selected_classes_str = shlex.quote(json.dumps(sc))
                evaluated_tasks_str = shlex.quote(json.dumps(evaluated_tasks))
                
                command1 = (
                    command +
                    f"{cl_hyper['training_mode']} "
                    f"{cl_hyper['cf_sol']} "
                    f"{cl_hyper['head_sol']} "
                    f"{cl_hyper['top_k']} "
                    f"{cl_hyper['high_lr']} "
                    f"{cl_hyper['low_lr']} "
                    f"{cl_hyper['t_criteria']} "
                    f"{cl_hyper['delta_w_interval']} "
                    f"{cl_hyper['heads_basis_t']} "
                    f"{selected_classes_str} "
                    f"{cl_hyper['n_tasks']} "
                    f"{evaluated_tasks_str} "
                    f"{cl_hyper['classes_per_task']} "
                    f"{cl_hyper['topk_lock']} "
                    f"{folder_id} "
                    f"{parent_f_id} "

                )
                result = subprocess.run(command1, shell=True, capture_output=False, text=True)
                print(result.stdout)
                if result.stderr:
                    print("Error:", result.stderr)
        else: 
            command1 = (
                command +
                f"{cl_hyper['training_mode']} "
                f"{cl_hyper['cf_sol']} "
                f"{cl_hyper['head_sol']} "
                f"{cl_hyper['top_k']} "
                f"{cl_hyper['high_lr']} "
                f"{cl_hyper['low_lr']} "
                f"{cl_hyper['t_criteria']} "
                f"{cl_hyper['delta_w_interval']} "
                f"{cl_hyper['heads_basis_t']} "
                f"{cl_hyper['topk_lock']} "
                f"{folder_id} "
                f"{parent_f_id} "
                
            )
            
            result = subprocess.run(command1, shell=True, capture_output=False, text=True)
            
            print(result.stdout)
            if result.stderr:
                print("Error:", result.stderr)
        if TEST:
            print("!!!! WARNING: BREAK OPERATION IS ON IN TESTING")
            break
            



# command = f"rm -rf -d /leonardo_work/{USER}/rcasciot/neuromodAI/SoftHebb-main/Training/results/hebb/result/network && mkdir /leonardo_work/{USER}/rcasciot/neuromodAI/SoftHebb-main/Training/results/hebb/result/network"
# result = subprocess.run(command, shell=True, capture_output=False, text=True)
    
# print(result.stdout)
# if result.stderr:
#     print("Error:", result.stderr)

if not folder_check(f"{parent_f_id}"):
    os.mkdir(f"/leonardo_work/{USER}/rcasciot/neuromodAI/SoftHebb-main/{parent_f_id}")
            

if data_num == 1: 
    command = f"cd /leonardo_work/{USER}/rcasciot/neuromodAI/batches/classes_CL/continual_learning && sbatch {dataset}.sh "
    all_classes = list(range(10))
    if dataset == "C100":
        all_classes = list(range(100))
    classes = []
    if n_tasks*classes_per_task > len(all_classes):
        for i in range(n_experiments):
            task_classes = []
            for j in range(n_tasks):
                task_classes.append(random.sample(all_classes, classes_per_task))
            classes.append(task_classes)
    else:
        for i in range(n_experiments):
            classes.append(random.sample(all_classes, classes_per_task*n_tasks))    

    if len(classes) > n_experiments:
        selected_classes = classes[:n_experiments]
    else: 
        n_experiments = len(classes)
        selected_classes = classes

    
    final = []
    #print("selected_classes: ", selected_classes)
    for el in selected_classes: 
        new = np.asarray(el)
        final.append(new.reshape(n_tasks,classes_per_task).tolist())
    print("final: ", final)
    if dataset == "C100": 
        dataset1 = "CIFAR100"
    elif dataset == "C10": 
        dataset1 = "CIFAR10"
    elif dataset == "IMG": 
        dataset1 = "ImageNette"
    elif dataset == "STL10": 
        dataset1 = "STL10"
    if folder_check(f"{parent_f_id}/TASKS_CL_{dataset1 +  folder_id}"):
        res = input(f"!!!! WARNING A FOLDER NAMED 'TASKS_CL_{ dataset +  folder_id}' already exits, press Y to continue anyways or N to abort: ")
        if res == "Y":
            execute_bash_command(evaluated_tasks, n_tasks, command, final)
    else:
        execute_bash_command(evaluated_tasks, n_tasks, command, final)
else: 
    command = f"cd /leonardo_work/{USER}/rcasciot/neuromodAI/batches/full_datasets_CL && sbatch {dataset}_{dataset2}.sh "
    if dataset == "C100": 
        dataset1 = "CIFAR100"
    elif dataset == "C10": 
        dataset1 = "CIFAR10"
    elif dataset == "IMG": 
        dataset1 = "ImageNette"
    elif dataset == "STL10": 
        dataset1 = "STL10"
    if dataset2 == "C100": 
        d2 = "CIFAR100"
    elif dataset2 == "C10": 
        d2 = "CIFAR10"
    elif dataset2 == "IMG": 
        d2 = "ImageNette"
    elif dataset2 == "STL10": 
        d2 = "STL10"
    if folder_check(f"{parent_f_id}/MULTD_CL_{dataset1 + '_' + folder_id + '_' + d2}"):
        res = input(f"!!!! WARNING A FOLDER NAMED 'MULTD_CL_{dataset1 + '_' + folder_id + '_' + d2}' already exits, press Y to continue anyways or N to abort: ")
        if res == "Y":
            execute_bash_command(evaluated_tasks=evaluated_tasks, n_tasks=n_tasks, command=command)
    else: 
        execute_bash_command(evaluated_tasks=evaluated_tasks, n_tasks=n_tasks, command=command)


