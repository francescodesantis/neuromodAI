import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from t_hyper import dataset, folder_id, parent_f_id, USER, n_tasks, classes_per_task, n_experiments


def read_data(file_path):
    # Load the entire content of the file into a single string
    with open(file_path, 'r') as file:
        content = file.read()

    # Parse the string into Python objects using eval
    data_dict = eval(content)
    wilcoxon_test = data_dict['wilcoxon_test']
    bootstrap_ci = data_dict['Bootstrap']
    performances = data_dict['Performances']
    return performances, wilcoxon_test, bootstrap_ci

def generate_latex_tables(performances, wilcoxon_test, bootstrap_ci, dataset):
    

    # Bootstrap CI LaTeX Table
    with open(f"/leonardo_work/{USER}/rcasciot/neuromodAI/SoftHebb-main/Tables/{dataset}_{n_tasks}T_{classes_per_task}C_p_ci.tex", 'w') as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{cccc}\n")
        f.write("\\toprule\n")
        f.write("Task & p-value & Lower CI & Upper CI \\\\\n")
        f.write("\\midrule\n")
        for idx in range(len(bootstrap_ci)):
            f.write(f"Task {idx} & {wilcoxon_test[idx][1]:.3f} &{bootstrap_ci[idx][0] :.2f} & {bootstrap_ci[idx][1] :.2f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{P-value calculated through the Wilcoxon test and Bootstrap Confidence Intervals between accuracies on evaluation of KPM-model and M-model. Regarding " + str(n_experiments)+ " experiments of incremental learning of "+str(n_tasks)+" tasks with "+str(classes_per_task)+" classes per task belonging to "+dataset+".}\n")
        f.write("\\end{table}\n")

    # Performance Metrics LaTeX Table
    with open(f"/leonardo_work/{USER}/rcasciot/neuromodAI/SoftHebb-main/Tables/{dataset}_{n_tasks}T_{classes_per_task}C_performance_metrics.tex", 'w') as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lc" + "c" * len(performances) + "}\n")
        f.write("\\toprule\n")
        
        sentence = "Task " 
        for k, v in performances.items():
            if k == "k=True, h=True":
                sentence += " & KPM-model"
            elif k == "k=False, h=True":
                sentence += " & M-model"
            elif k == "k=True, h=False":
                sentence += " & KP-model"
            elif k == "k=False, h=False":
                sentence += " & V-model"
        sentence += " & $Acc_{KPM}-Acc_{M}$"
        f.write(sentence + " \\\\\n")
        f.write("\\midrule\n")
        h_line = False
        for idx in range(len(list(performances.values())[0])): 
            measure = list(performances.values())[0][idx][0]
            values = [performances[k][idx][1] for k in performances]
            max_value = max(values)
            sentence = ""
            if not h_line and "Eval" in measure: 
                    sentence += "\n \hline \n"
                    h_line = True
            f.write(sentence+measure)
            sentence = ""
            for value in values:
                
                if value == max_value and "R" not in measure:
                    sentence += f" & \\textbf{{{value:.2f}}}"
                else:
                    sentence += f" & {value:.2f}"
                
            if "Eval" in measure:
                sentence += f" & {(values[3]-values[2]):.2f}"
            else: 
                sentence += f" & $-$"

            f.write(f"{sentence} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Accuracy results on incremental learning of "+str(n_tasks)+" tasks with "+str(classes_per_task)+" classes per task belonging to "+dataset+ " on "+str(n_experiments)+" experiments.}\n")
        f.write("\\end{table}\n")
    
    avg_acc = {}

    for k, v in performances.items():
        tot = 0
        for res in v:
            if "Eval" in res[0]:
                tot += res[1]
        avg_acc[k] = tot/n_tasks
    print(avg_acc)

# Main function to orchestrate the file creation
    # Adjust the path as needed
if dataset == "C100": 
    dataset = "CIFAR100"
elif dataset == "C10": 
    dataset = "CIFAR10"
performances, wilcoxon_test, bootstrap_ci = read_data(f"/leonardo_work/{USER}/rcasciot/neuromodAI/SoftHebb-main/{parent_f_id}/TASKS_CL_{dataset+ folder_id}/TASKS_CL_{dataset+ folder_id}_statistics.txt")
generate_latex_tables(performances, wilcoxon_test, bootstrap_ci, dataset)

