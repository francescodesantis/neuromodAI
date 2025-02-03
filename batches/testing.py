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

def execute_bash_command(classes: list, command: str):
    modes = ["successive", "consecutive", "simultaneous"]
    lrs = [(0.0, 1.0), (2000, 1.0), (0.2, 0.8)]
    sols = [(True, True), (False, False), (True, False), (False, True)]
    topks = [0.1, 0.2, 0.5, 0.7, 0.85, 0.9, 1.0]
    delta_w_intervals = [20, 100, 300]
    path = '/leonardo_work/try24_antoniet/rcasciot/neuromodAI/batches/classes_CL/continual_learning/input.json'
    lr = lrs[2]
    mode = modes[1]
    
    for sol in sols:
        for sc in classes:
            cl_hyper = {
                'training_mode': mode,
                'cf_sol': sol[0],
                'head_sol': sol[1],
                'top_k': topks[3],
                'high_lr': lr[0],
                'low_lr': lr[1],
                't_criteria': 'mean',
                'delta_w_interval': delta_w_intervals[0],
                'heads_basis_t': 0.60,
                'selected_classes': [[2,8], [1,5]]

            }

            # f = open('/leonardo_work/try24_antoniet/rcasciot/neuromodAI/batches/classes_CL/continual_learning/input.json', "r")
            # cl_hyper = json.load(f)
           
            print(cl_hyper['selected_classes'])
            selected_classes_str = shlex.quote(json.dumps(cl_hyper['selected_classes']))
            
            #command = f"cd /leonardo_work/try24_antoniet/rcasciot/neuromodAI/batches/classes_CL/continual_learning && sbatch C100_2C.sh {cl_hyper['training_mode']} {cl_hyper['cf_sol']} {cl_hyper['head_sol']} {cl_hyper['top_k']} {cl_hyper['high_lr']} {cl_hyper['low_lr']} {cl_hyper['t_criteria']} {cl_hyper['delta_w_interval']} {cl_hyper['heads_basis_t']} {selected_classes_str} "
            command1 = (
                "cd /leonardo_work/try24_antoniet/rcasciot/neuromodAI/batches/classes_CL/continual_learning && "
                "sbatch C10_2C.sh "
                f"{cl_hyper['training_mode']} "
                f"{cl_hyper['cf_sol']} "
                f"{cl_hyper['head_sol']} "
                f"{cl_hyper['top_k']} "
                f"{cl_hyper['high_lr']} "
                f"{cl_hyper['low_lr']} "
                f"{cl_hyper['t_criteria']} "
                f"{cl_hyper['delta_w_interval']} "
                f"{cl_hyper['heads_basis_t']} "
                f"{selected_classes_str}"
            )
            
            result = subprocess.run(command1, shell=True, capture_output=False, text=True)
            
            print(result.stdout)
            if result.stderr:
                print("Error:", result.stderr)
            break
        break
            

# Example usage
num_classes = 2
command = f"cd /leonardo_work/try24_antoniet/rcasciot/neuromodAI/batches/classes_CL/continual_learning && sbatch --wrap=C100_2C.sh"
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
res = list(itertools.permutations(classes, num_classes*2))
#res = list(itertools.permutations(res, num_classes))
random.shuffle(res)
selected_classes = res
if len(res) > 50:
    selected_classes = res[:51]

final = []
for el in selected_classes: 
    new = np.asarray(el)
    final.append(new.reshape(2,2).tolist())
#print(final)
execute_bash_command(final, command)
