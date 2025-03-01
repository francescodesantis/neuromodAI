import json
import os
import uuid
import shutil

TEST = True

classes_per_task = 2
n_experiments = 80
n_tasks = 9
evaluated_tasks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ]

if TEST: 
    n_experiments = 1
    n_tasks = 5
    

data_num = 1 # set to 2 to use in multi dataset CL mode, otherwise to 1 for tasks from the same dataset.
dataset="C100"
dataset2 = "C10"
folder_id = f"_d1_{n_tasks}tasks"
if data_num == 1:
    parent_f_id = f"experiments/EXP_{dataset}_{classes_per_task}C_test"
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
                    'delta_w_interval': 50,
                    'heads_basis_t': 0.90,
                    'n_tasks': n_tasks, 
                    'classes_per_task': classes_per_task
                }

