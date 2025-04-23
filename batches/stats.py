import os
import json
import numpy as np
import matplotlib.pyplot as plt
from t_hyper import classes_per_task, n_experiments, n_tasks, dataset, evaluated_tasks, folder_id, cl_hyper, parent_f_id, data_num, dataset2, USER

import numpy as np
from scipy.stats import wilcoxon, kruskal, ttest_rel


def paired_t_test(accuracies_continual, accuracies_baseline):
    """
    Perform a paired t-test between the continual learning solution and the baseline.
    
    Parameters:
      accuracies_continual (array-like): Accuracies with continual learning active.
      accuracies_baseline (array-like): Accuracies without continual learning.
      
    Returns:
      t_stat (float): The t-statistic.
      p_value (float): The p-value from the test.
    """
    t_stat, p_value = ttest_rel(accuracies_continual, accuracies_baseline)
    return t_stat, p_value

def bootstrap_confidence_interval(data, num_bootstrap=10000, confidence=0.95):
    """
    Compute the bootstrap confidence interval for the mean of the data.
    
    Parameters:
      data (array-like): The data vector (e.g., accuracies from one condition).
      num_bootstrap (int): Number of bootstrap samples (default 10,000).
      confidence (float): Confidence level (default 0.95).
      
    Returns:
      (lower, upper): Tuple with lower and upper bounds of the confidence interval.
    """
    data = np.array(data)
    means = []
    n = len(data)
    for _ in range(num_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - confidence) / 2 * 100)
    upper = np.percentile(means, (confidence + (1 - confidence) / 2) * 100)
    return lower, upper

def bootstrap_difference_ci(accuracies_continual, accuracies_baseline, num_bootstrap=10000, confidence=0.95):
    """
    Compute the bootstrap confidence interval for the difference in means 
    between the continual learning solution and the baseline.
    
    Parameters:
      accuracies_continual (array-like): Accuracies with continual learning active.
      accuracies_baseline (array-like): Accuracies without continual learning.
      num_bootstrap (int): Number of bootstrap samples (default 10,000).
      confidence (float): Confidence level (default 0.95).
      
    Returns:
      (lower, upper): Tuple with lower and upper bounds of the confidence interval of the difference.
    """
    accuracies_continual = np.array(accuracies_continual)
    accuracies_baseline = np.array(accuracies_baseline)
    differences = []
    n = len(accuracies_continual)
    for _ in range(num_bootstrap):
        indices = np.random.randint(0, n, n)
        diff_sample = np.mean(accuracies_continual[indices]) - np.mean(accuracies_baseline[indices])
        differences.append(diff_sample)
    lower = np.percentile(differences, (1 - confidence) / 2 * 100)
    upper = np.percentile(differences, (confidence + (1 - confidence) / 2) * 100)
    return lower, upper

def cohen_d_paired(accuracies_continual, accuracies_baseline):
    """
    Compute Cohen's d for paired samples (Cohen's dz).
    
    Parameters:
    x (array-like): Accuracy scores of model 1 on the same dataset.
    y (array-like): Accuracy scores of model 2 on the same dataset.
    
    Returns:
    float: Cohen's dz (effect size for paired samples).
    """
    x = np.array(accuracies_continual)
    y = np.array(accuracies_baseline)
    
    # Compute the differences
    d = x - y
    
    # Mean of the differences
    mean_d = np.mean(d)
    
    # Standard deviation of the differences
    std_d = np.std(d, ddof=1)  # ddof=1 for sample standard deviation
    
    # Cohen's dz
    if std_d == 0:
        return 0  # Avoid division by zero; means are identical
    d_z = mean_d / std_d
    
    return d_z


    

def create_boxplot_graph_runs(stats):
    fig, axes = plt.subplots(n_tasks, n_tasks, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, res in enumerate(stats):
        ax = axes[idx]
        performances = res['performances']
        
        r_keys = sorted(performances.keys(), key=lambda r: int(r[1:]))
        test_accs = [performances[r]['test_acc'] for r in r_keys]
        std_accs = [performances[r]['std_test_acc'] for r in r_keys]
        
        ax.boxplot([test_accs], vert=True, patch_artist=True)
        ax.set_title(f"cf_sol={res['cl_hyper']['cf_sol']}, head_sol={res['cl_hyper']['head_sol']}")
        ax.set_xticklabels(["Test Accuracy"])
        ax.set_ylabel("Accuracy")
        
    plt.suptitle("Boxplot of Standard Deviations for Different Configurations of Training")
    plt.tight_layout()
    if data_num == 1:
        plt.savefig(f"/leonardo_work/{USER}/rcasciot/neuromodAI/SoftHebb-main/{parent_f_id}/TASKS_CL_{dataset+ folder_id}/TASKS_CL_{dataset+ folder_id}_STD-runs", bbox_inches='tight')
    else: 
        plt.savefig(f"/leonardo_work/{USER}/rcasciot/neuromodAI/SoftHebb-main/{parent_f_id}/MULTD_CL_{dataset + '_' + dataset2  + '_' + folder_id}/MULTD_CL_{dataset + '_' + dataset2  + '_' + folder_id}_STD-runs", bbox_inches='tight')

    plt.close()

def create_boxplot_graph_eval(stats):
   
    fig, axes = plt.subplots(1, n_tasks, figsize=(70, 10))
    axes = axes.flatten()
    keys = stats[0]["eval_raw_stats"].keys()
    index = 0
    for eval_k in keys: 
        evals = []
        labels = []
        for i in range(len(stats)):
            evals.append(stats[i]["eval_raw_stats"][eval_k])
            labels.append(f"cf_sol={stats[i]['cl_hyper']['cf_sol']},\nhead_sol={stats[i]['cl_hyper']['head_sol']}")
        
        ax = axes[index]
        ax.boxplot(evals, vert=True, patch_artist=True)
        ax.set_xticklabels(labels)
        ax.set_title(eval_k)
        ax.set_ylabel("Accuracy")
        index += 1

        
    plt.suptitle("Boxplot of Standard Deviations for Different Configurations of evaluation")
    plt.tight_layout()
    if data_num == 1:
        plt.savefig(f"/leonardo_work/{USER}/rcasciot/neuromodAI/SoftHebb-main/{parent_f_id}/TASKS_CL_{dataset+ folder_id}/TASKS_CL_{dataset+ folder_id}_STD-eval", bbox_inches='tight')
    else:
        plt.savefig(f"/leonardo_work/{USER}/rcasciot/neuromodAI/SoftHebb-main/{parent_f_id}/MULTD_CL_{dataset + '_' + dataset2  + '_' + folder_id}/MULTD_CL_{dataset + '_' + dataset2  + '_' + folder_id}_STD-eval", bbox_inches='tight')

    plt.close()



def wilcoxon_signed_rank_test(model1_accuracies, model2_accuracies):
    """
    Perform the Wilcoxon Signed-Rank Test to compare two versions of the same model.
    
    Parameters:
        model1_accuracies (list or np.array): Accuracy values of the first model version.
        model2_accuracies (list or np.array): Accuracy values of the second model version.
    
    Returns:
        T-statistic and p-value
    """
    T_statistic, p_value = wilcoxon(model1_accuracies, model2_accuracies)
    return T_statistic, p_value


# Function to compute average accuracy and standard deviation
def average_behavior(dataset, n_experiments, classes_per_task, n_tasks, path):
    tot_test_acc = {}
    tot_test_loss = {}
    std_test_acc = {}
    std_test_loss = {}
    performances = {}
    count = 0

    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            if ".json" not in file:
                continue
            with open(os.path.join(root, file), "r") as f:
                json_obj = json.load(f)
                obj = json_obj["cl_hyper"]
                
               # print(dataset, json_obj["R0"]['dataset_sup']["name"])
                if dataset == json_obj["R0"]['dataset_sup']["name"] and cl_hyper["cf_sol"] == obj["cf_sol"] and cl_hyper["head_sol"] == obj["head_sol"]:
                    count += 1
                    #print("cl_hyper: ", len(cl_hyper))
                    #print("cl_hyper_obj: ", len(cl_hyper_obj))

                    #print("#####################################")

                    for k in json_obj.keys():
                        if "eval" in k:
                            if k not in tot_test_acc:
                                tot_test_acc[k] = [json_obj[k]["test_acc"]]
                                tot_test_loss[k] = [json_obj[k]["test_loss"]]
                            else:
                                tot_test_acc[k].append(json_obj[k]["test_acc"])
                                tot_test_loss[k].append(json_obj[k]["test_loss"])

                        elif "R" in k:
                            if k not in performances:
                                performances[k] = {"test_acc": [json_obj[k]["test_acc"]], "test_loss": [json_obj[k]["test_loss"]]}
                            else:
                                performances[k]["test_acc"].append(json_obj[k]["test_acc"])
                                performances[k]["test_loss"].append(json_obj[k]["test_loss"])
    performances_ret = performances.copy()
    tot_test_acc_ret = tot_test_acc.copy()
    # print("performances: ", performances_ret)
    # print("tot_test_acc: ", tot_test_acc)

    # Compute means and standard deviations
    for k in tot_test_acc.keys():
        tmp = tot_test_acc[k]

        tot_test_acc[k] = np.mean(tot_test_acc[k])
        std_test_acc[k] = np.std(tmp)

        tmp = tot_test_loss[k]
        tot_test_loss[k] = np.mean(tot_test_loss[k])
        std_test_loss[k] = np.std(tmp)

    for k in performances.keys():
        tmp = performances[k]["test_acc"]
        tmp1 = performances[k]["test_loss"]

        performances[k]["test_acc"] = np.mean(performances[k]["test_acc"])
        performances[k]["test_loss"] = np.mean(performances[k]["test_loss"])

        performances[k]["std_test_acc"] = np.std(tmp)
        performances[k]["std_test_loss"] = np.std(tmp1)

    return {
        "dataset": dataset,
        "n_experiments": count,
        "n_tasks": n_tasks,
        "classes_per_task": classes_per_task,
        "avg_test_acc": tot_test_acc,
        "std_test_acc": std_test_acc,
        "cl_hyper": cl_hyper.copy(),
        "performances": performances,
        "count": count,
        "eval_raw_stats": tot_test_acc_ret, 
        "run_raw_stats": performances_ret
    }

sols = [(False, False), (True, False), (False, True),(True, True)]
stats = []
counter = 0
important_stats = []
eval_stats = []
run_stats = []


for sol in sols:
   
    cl_hyper['cf_sol'] = sol[0]
    cl_hyper['head_sol'] = sol[1]
    
    

    if dataset == "C100": 
        dataset = "CIFAR100"
    elif dataset == "C10": 
        dataset = "CIFAR10"
    elif dataset == "IMG": 
        dataset = "ImageNette"
    elif dataset == "STL10": 
        dataset = "STL10"
    if dataset2 == "C100": 
        dataset2 = "CIFAR100"
    elif dataset2 == "C10": 
        dataset2 = "CIFAR10"
    elif dataset2 == "IMG": 
        dataset2 = "ImageNette"
    elif dataset2 == "STL10": 
        dataset2 = "STL10"

    if data_num == 1:
        res = average_behavior(dataset, n_experiments, classes_per_task, n_tasks, f"/leonardo_work/{USER}/rcasciot/neuromodAI/SoftHebb-main/{parent_f_id}/TASKS_CL_{dataset+ folder_id}")
    else:
        res = average_behavior(dataset, n_experiments, classes_per_task, n_tasks, f"/leonardo_work/{USER}/rcasciot/neuromodAI/SoftHebb-main/{parent_f_id}/MULTD_CL_{dataset + '_' + dataset2  + '_' + folder_id}")

    print(res)
    stats.append(res)
     
    
    
print("important_stats: ", important_stats)
wilcoxon_test = []
paired_test = []
bootstrap_CI = []
bootstrap_difference_CI = []
cohen = []
for key in stats[3]["eval_raw_stats"].keys(): 
    print(len(stats[0]["eval_raw_stats"][key]), len(stats[3]["eval_raw_stats"][key]))
    wilcoxon_test.append(wilcoxon_signed_rank_test(stats[3]["eval_raw_stats"][key], stats[2]["eval_raw_stats"][key]))
    paired_test.append(paired_t_test(stats[3]["eval_raw_stats"][key], stats[2]["eval_raw_stats"][key]))
    bootstrap_CI.append(bootstrap_confidence_interval(stats[3]["eval_raw_stats"][key]))
    bootstrap_difference_CI.append(bootstrap_difference_ci(stats[3]["eval_raw_stats"][key], stats[2]["eval_raw_stats"][key]))
    cohen.append(cohen_d_paired(stats[3]["eval_raw_stats"][key], stats[2]["eval_raw_stats"][key]))

print("\n\n ######################################################################################\n\n")
print("wilcoxon_test: ", wilcoxon_test)

print("Paired t-test: ", paired_test)
    
    # Bootstrap confidence interval for the mean of the continual learning accuracies
print("Bootstrap CI for continual learning mean: ",bootstrap_CI )
    
    # Bootstrap confidence interval for the difference in means
print("Bootstrap CI for difference in means: ",  bootstrap_difference_CI)
    
    # Cohen's d effect size
print("Cohen's d effect size: ", cohen)
print("\n\n ######################################################################################\n\n")

# Create the plot
plt.figure(figsize=(30, 20))
plt.rcParams.update({'font.size': 25})

annotations = {}
accuracies = {}
# print("STATS: ", stats)

for r in stats:
    performances = r['performances']
    avg_test_acc = r["avg_test_acc"]
    std_test_acc = r["std_test_acc"]

    r_keys = sorted(performances.keys(), key=lambda r: int(r[1:]))
    test_accs = [performances[r]['test_acc'] for r in r_keys]
    std_accs = [performances[r]['std_test_acc'] for r in r_keys]

    # Append avg_test_acc values
    test_accs += [avg_test_acc[r] for r in avg_test_acc.keys()]
    std_accs += [std_test_acc[r] for r in std_test_acc.keys()]
    r_keys += list(avg_test_acc.keys())
    print("LINE :", r_keys)
    new_rkeys = []
    for lab in r_keys: 
        if "eval" in lab: 
            new_rkeys.append(f"Eval on T{lab.split('_')[1]}")
        else:
            new_rkeys.append(lab)
    r_keys = new_rkeys
    #label = plt.text(0.50, 0.02, f"kernel solution={r['cl_hyper']['cf_sol']}, head solution={r['cl_hyper']['head_sol']}", horizontalalignment='left', wrap=True ,)
    label = f"k={r['cl_hyper']['cf_sol']}, h={r['cl_hyper']['head_sol']}"

    xs = list(range(len(r_keys)))
    line, = plt.plot(xs, test_accs, marker='o', label=label)
    accuracies[label] = []
    for i in range(len(xs)):
        accuracies[label].append((r_keys[i], test_accs[i] ))
    
    color = line.get_color()
    stat_annotations = {}
    for x, (acc, std) in enumerate(zip(test_accs, std_accs)):
        annotations.setdefault(x, []).append((acc, std, color))
        stat_annotations.setdefault(x, []).append((acc, std, color, label))
print("ANNOTATIONS: ", annotations)
plt.xticks(xs, r_keys)
for x, ann_list in annotations.items():
    sorted_ann = sorted(ann_list, key=lambda tup: tup[0])
    n = len(sorted_ann)
    center = sum(a for a, s, c in sorted_ann) / n
    spacing = 1
    start_offset = -spacing * (n - 1) / 2
    for i, (orig_acc, std, color) in enumerate(sorted_ann):
        new_y = center + start_offset + i * spacing
        plt.text(x + 0.1, new_y, f'{orig_acc:.1f}', color=color, ha='left', va='center', fontsize=18)

plt.xlabel('R#')
plt.ylabel('Test Accuracy')

plt.title(f"{dataset} with {classes_per_task} classes per task, {n_tasks} tasks per experiment, on {res['count']} experiments")
p_values = ""
for i in range(len(wilcoxon_test)):
    p_values += f"- evaluation on task {i}: {wilcoxon_test[i][1]}\n"
#text = plt.text(0.50, 0.02, f'P-values between evaluations having both solutions on and just the head solution on:\n {p_values}', horizontalalignment='left', wrap=True ) 
statistics = "{'wilcoxon_test': " + str(wilcoxon_test) +",\n" +"'Bootstrap':" + str(bootstrap_difference_CI)
statistics += ",\n'Performances': " + json.dumps(accuracies, indent=4) + "}"
plt.legend()
plt.grid(True)
plt.tight_layout()
if data_num == 1: 
    plt.savefig(f"/leonardo_work/{USER}/rcasciot/neuromodAI/SoftHebb-main/{parent_f_id}/TASKS_CL_{dataset+ folder_id}/TASKS_CL_{dataset+ folder_id}", bbox_inches='tight')
    with open(f"/leonardo_work/{USER}/rcasciot/neuromodAI/SoftHebb-main/{parent_f_id}/TASKS_CL_{dataset+ folder_id}/TASKS_CL_{dataset+ folder_id}_statistics.txt", "w") as f: 
        f.write(statistics)

else:
    plt.savefig(f"/leonardo_work/{USER}/rcasciot/neuromodAI/SoftHebb-main/{parent_f_id}/MULTD_CL_{dataset + '_' + dataset2  + '_' + folder_id}/MULTD_CL_{dataset + '_' + dataset2  + '_' + folder_id}", bbox_inches='tight')
    with open(f"/leonardo_work/{USER}/rcasciot/neuromodAI/SoftHebb-main/{parent_f_id}/MULTD_CL_{dataset + '_' + dataset2  + '_' + folder_id}/MULTD_CL_{dataset + '_' + dataset2  + '_' + folder_id}_statistics.txt", "w") as f: 
        f.write(statistics)


plt.close()

create_boxplot_graph_eval(stats)
create_boxplot_graph_runs(stats)