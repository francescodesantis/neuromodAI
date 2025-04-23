def average_accuracy(eval_acc):
    """
    Compute the Average Accuracy (AA) after learning all tasks.
    This is simply the mean of final evaluation accuracies of all tasks.
    
    Parameters:
        eval_acc (list or array): length T, final accuracies for each task (a_{T,j} for j=1..T).
    Returns:
        float: Average Accuracy over all T tasks.
    """
    # Ensure eval_acc is a list or array of numbers
    accuracies = list(eval_acc)
    T = len(accuracies)
    if T == 0:
        return 0.0
    # Average of final accuracies
    AA = sum(accuracies) / T
    return AA

def average_incremental_accuracy(train_acc, eval_acc, acc_matrix=None):
    """
    Compute the Average Incremental Accuracy (AIA) after learning all tasks.
    
    If an accuracy matrix is provided (acc_matrix), it uses the exact definition:
      AIA = (1/T) * sum_{i=1..T} AA_i,
    where AA_i = (1/i) * sum_{j=1..i} a_{i,j}.
    The acc_matrix should be a 2D list/array where acc_matrix[i][j] = a_{i+1, j+1},
    i.e. accuracy on task (j+1) after training task (i+1).
    
    If acc_matrix is not provided, the function will approximate AIA using the given vectors:
    It assumes that after each task, previously learned tasks have accuracy equal to their final accuracy (a pessimistic assumption).
    Under this assumption:
      For each i from 1 to T:
         For task j <= i:
             use accuracy = train_acc[j-1] if j == i (just trained task),
                        or eval_acc[j-1] if j < i (approximate accuracy of old task after i-th training).
         Compute AA_i from these, then average all AA_i.
    
    Parameters:
        train_acc (list or array): length T, accuracy on each task when it was trained (a_{j,j}).
        eval_acc  (list or array): length T, final accuracy on each task after all tasks (a_{T,j}).
        acc_matrix (list of lists or 2D array, optional): full accuracy matrix for exact computation (if available).
    Returns:
        float: Average Incremental Accuracy.
    """
    T = len(train_acc)
    # If full accuracy matrix is provided, use it for exact computation of AIA
    if acc_matrix is not None:
        # Compute AA_i for each i using the matrix, then average them
        AAs = []
        for i in range(T):
            # i tasks means index 0..i in 0-based indexing
            num_tasks = i + 1
            # accuracy for tasks 1..num_tasks after training task num_tasks:
            # these are acc_matrix[num_tasks-1][0..num_tasks-1]
            accs_after_i = acc_matrix[i][:num_tasks]
            AA_i = sum(accs_after_i) / num_tasks
            AAs.append(AA_i)
        return sum(AAs) / T  # average of AA_1..AA_T
    
    # Otherwise, approximate using train_acc and eval_acc
    train_acc = list(train_acc)
    eval_acc = list(eval_acc)
    if T == 0:
        return 0.0
    AAs = []
    for i in range(1, T+1):  # i = number of tasks learned so far
        # For each task j <= i, determine accuracy after learning i tasks
        acc_sum = 0.0
        for j in range(1, i+1):
            if j == i:
                # accuracy on the i-th task right after training it (from train_acc)
                acc_sum += train_acc[j-1]
            else:
                # for previous task j (j < i), approximate its accuracy after i-th task as its final accuracy
                acc_sum += eval_acc[j-1]
        AA_i = acc_sum / i
        AAs.append(AA_i)
    # Average of all AA_i
    return sum(AAs) / T

def backward_transfer(train_acc, eval_acc):
    """
    Compute the Backward Transfer (BWT) after learning all tasks.
    BWT = (1/(T-1)) * sum_{j=1}^{T-1} (a_{T,j} - a_{j,j}),
    i.e. average influence of learning new tasks on old tasks' performance.
    
    A negative result indicates overall forgetting of past tasks.
    If T < 2, returns 0.0 (no past task to compare).
    
    Parameters:
        train_acc (list or array): length T, accuracy on task j when it was trained (a_{j,j}).
        eval_acc  (list or array): length T, final accuracy on task j after all tasks (a_{T,j}).
    Returns:
        float: Backward Transfer value (positive, zero, or negative).
    """
    train_acc = list(train_acc)
    eval_acc = list(eval_acc)
    T = len(train_acc)
    if T < 2:
        return 0.0  # not defined for fewer than 2 tasks
    # Sum differences for tasks 1..T-1
    diff_sum = 0.0
    for j in range(1, T):  # j = 1 to T-1 (using 1-based task index)
        diff = eval_acc[j-1] - train_acc[j-1]  # a_{T,j} - a_{j,j}
        diff_sum += diff
    BWT = diff_sum / (T - 1)
    return BWT

def forgetting_measure(train_acc, eval_acc):
    """
    Compute the Forgetting Measure (FM) after learning all tasks.
    We calculate forgetting for each task j as the drop from its training-time accuracy to its final accuracy: f_j = a_{j,j} - a_{T,j}.
    (This assumes the highest accuracy for task j was when it was trained.)
    Then FM = (1/(T-1)) * sum_{j=1}^{T-1} f_j.
    
    If T < 2, returns 0.0 (no forgetting since there is no past task).
    
    Parameters:
        train_acc (list or array): length T, accuracy on task j when trained (a_{j,j}).
        eval_acc  (list or array): length T, final accuracy on task j after all tasks (a_{T,j}).
    Returns:
        float: Forgetting measure (average drop in accuracy for old tasks).
    """
    train_acc = list(train_acc)
    eval_acc = list(eval_acc)
    T = len(train_acc)
    if T < 2:
        return 0.0
    drop_sum = 0.0
    for j in range(1, T):  # tasks 1 to T-1
        drop = train_acc[j-1] - eval_acc[j-1]  # (a_{j,j} - a_{T,j})
        if drop < 0:
            drop = 0.0  # if negative (shouldn't happen if a_{j,j} was max), clip to 0 as no forgetting
        drop_sum += drop
    FM = drop_sum / (T - 1)
    return FM

def forward_transfer(train_acc, eval_acc, base_acc=None):
    """
    Compute the Forward Transfer (FWT) after learning all tasks.
    FWT = (1/(T-1)) * sum_{j=2}^{T} (a_{j,j} - \tilde{a}_j),
    where a_{j,j} is accuracy on task j after learning it in sequence, 
    and \tilde{a}_j is the baseline accuracy on task j without prior tasks.
    
    This function uses train_acc for a_{j,j}. For baseline \tilde{a}_j, the `base_acc` list can be provided.
    If base_acc is None, we assume \tilde{a}_j = 0 for all j (no prior knowledge baseline).
    If T < 2, returns 0.0 (FWT is not defined for only one task).
    
    Parameters:
        train_acc (list or array): length T, accuracy on task j when trained (a_{j,j}).
        eval_acc  (list or array): length T, final accuracy on task j after all tasks (not used in FWT calculation here).
        base_acc  (list or array, optional): length T, baseline accuracy for each task j (\\tilde{a}_j).
                                            If not provided, baseline is assumed 0 for all tasks >1.
    Returns:
        float: Forward Transfer value.
    """
    train_acc = list(train_acc)
    T = len(train_acc)
    if T < 2:
        return 0.0
    if base_acc is None:
        # Assume zero baseline for tasks (meaning no prior knowledge yields 0 accuracy).
        base_acc = [0.0] * T
    else:
        base_acc = list(base_acc)
        # If baseline list is shorter than T or missing first task, pad with 0 for safety
        if len(base_acc) < T:
            base_acc = base_acc + [0.0] * (T - len(base_acc))
    # Compute sum of (a_{j,j} - base_j) for j = 2..T (index 1..T-1 in 0-based)
    sum_gain = 0.0
    for j in range(2, T+1):  # j = 2 to T
        seq_perf = train_acc[j-1]        # a_{j,j}
        baseline_perf = base_acc[j-1]    # \tilde{a}_j
        gain = seq_perf - baseline_perf
        sum_gain += gain
    FWT = sum_gain / (T - 1)
    return FWT

import numpy as np
import matplotlib.pyplot as plt

def read_statistics_file(file_path):
    """
    Reads a statistics.txt file and extracts Cohen's d, Wilcoxon p-values, and Bootstrap CI.
    
    Parameters:
        file_path (str): Path to the statistics.txt file.
    
    Returns:
        dict: A dictionary containing extracted data for plotting.
    """
    data = {
        "tasks": [],
        "cohen_d": [],
        "wilcoxon_p": [],
        "ci_diff_lower": [],
        "ci_diff_upper": []
    }
    
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            # Cohen's d
            if "Cohen's d effect size" in line:
                values = line.split(":")[1].strip().strip("[]").split(",")
                data["cohen_d"] = [float(v) for v in values]
            
            # Wilcoxon test p-values
            elif "wilcoxon_test" in line:
                values = line.split(":")[1].strip().strip("[]").split("), (")
                p_values = [float(v.split(", ")[1]) for v in values]
                data["wilcoxon_p"] = p_values
            
            # Bootstrap CI for difference in means
            elif "Bootstrap CI for difference in means" in line:
                values = line.split(":")[1].strip().strip("[]").split("), (")
                lower_bounds = [float(v.split(", ")[0].strip("(")) for v in values]
                upper_bounds = [float(v.split(", ")[1].strip(")")) for v in values]
                data["ci_diff_lower"] = lower_bounds
                data["ci_diff_upper"] = upper_bounds
    
    # Define tasks dynamically based on extracted data length
    data["tasks"] = [f"Task {i}" for i in range(len(data["cohen_d"]))]
    
    return data

def plot_wilcoxon(data):
    """Plots Wilcoxon test p-values on a log scale."""
    plt.figure(figsize=(10, 5))
    plt.plot(data["tasks"], data["wilcoxon_p"], marker='o', linestyle='-', color='b', label="Wilcoxon P-Value")
    plt.axhline(y=0.05, color='r', linestyle='--', label="Significance Threshold (0.05)")
    plt.yscale('log')  # Log scale to emphasize small p-values
    plt.ylabel("P-Value (log scale)")
    plt.xlabel("Task")
    plt.title("Wilcoxon Test P-Values Across Tasks")
    plt.legend()
    plt.grid()
    plt.show()

def plot_cohen_d(data):
    """Plots Cohen's d effect sizes."""
    plt.figure(figsize=(10, 5))
    plt.bar(data["tasks"], data["cohen_d"], color='g', alpha=0.7)
    plt.axhline(y=0.2, color='orange', linestyle='--', label="Small Effect (0.2)")
    plt.axhline(y=0.5, color='r', linestyle='--', label="Medium Effect (0.5)")
    plt.axhline(y=0, color='black', linestyle='--', label="No Effect")
    plt.ylabel("Cohen's d (Effect Size)")
    plt.xlabel("Task")
    plt.title("Effect Sizes (Cohen's d) per Task")
    plt.legend()
    plt.grid()
    plt.show()

def plot_bootstrap_ci(data):
    """Plots Bootstrap Confidence Intervals for Difference in Means."""
    plt.figure(figsize=(10, 5))
    for i, task in enumerate(data["tasks"]):
        plt.plot([task, task], [data["ci_diff_lower"][i], data["ci_diff_upper"][i]], marker="_", markersize=10, color='g')
    plt.axhline(y=0, color='r', linestyle='--', label="No Difference Line")
    plt.ylabel("Difference in Means")
    plt.xlabel("Task")
    plt.title("Bootstrap Confidence Intervals for Difference in Means")
    plt.legend()
    plt.grid()
    plt.show()

# Main execution
file_path = "statistics.txt"  # Change this if your file is located elsewhere
data = read_statistics_file(file_path)

# Generate plots
plot_wilcoxon(data)
plot_cohen_d(data)
plot_bootstrap_ci(data)
