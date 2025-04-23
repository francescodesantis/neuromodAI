import json
import os
import subprocess
import uuid
import shutil

TEST = False
USER = 'IscrC_CATASTRO'
classes_per_task = 2
n_experiments = 80
n_tasks = 6

evaluated_tasks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ]
#neuromodAI/SoftHebb-main/experiments/EXP_C100_4C/TASKS_CL_CIFAR100_d3_6tasks
#neuromodAI/SoftHebb-main/experiments/EXP_C100_2C/TASKS_CL_CIFAR100_c1_big_6tasks
if TEST: 
    n_experiments = 80
    #n_tasks = 5
    

data_num = 1 # set to 2 to use in multi dataset CL mode, otherwise to 1 for tasks from the same dataset.
dataset="C100"
dataset2 = "C10"
folder_id = f"_d1_{n_tasks}tasks"
if data_num == 1:
    parent_f_id = f"experiments/EXP_{dataset}_{classes_per_task}C"
else:
    parent_f_id = f"experiments/EXP_{dataset}_{dataset2}"

# C100, C10, STL10, IMG

cl_hyper = {
                    'training_mode': 'consecutive',
                    'top_k': 0.15,
                    'topk_lock': False,
                    'high_lr': 0.15,
                    'low_lr': 1,
                    't_criteria': 'activations', # KSE or activations
                    'delta_w_interval': 30,
                    'heads_basis_t': 0.90,
                    'n_tasks': n_tasks, 
                    'classes_per_task': classes_per_task
                }

# for root, dirs, files in os.walk("/leonardo_work/IscrC_CATASTRO/rcasciot/neuromodAI/SoftHebb-main/experiments/EXP_C10_2C/TASKS_CL_CIFAR10_a1_8tasks", topdown=False):
#         for file in files:
#             if ".json" not in file:
#                 continue
#             with open(os.path.join(root, file), "r") as f:
#                 json_obj = json.load(f)
                
                
#                # print(dataset, json_obj["R0"]['dataset_sup']["name"])
#                 if "b4" not in list(json_obj["model_config"].keys()): 
#                     result = subprocess.run(f"rm -rf /leonardo_work/IscrC_CATASTRO/rcasciot/neuromodAI/SoftHebb-main/experiments/EXP_C10_2C/TASKS_CL_CIFAR10_a1_8tasks/{file}", shell=True, capture_output=False, text=True)
#                     print(result.stdout)
#                     if result.stderr:
#                         print("Error:", result.stderr)
                    