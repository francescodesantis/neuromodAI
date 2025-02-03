from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import numpy as np
import os 

import json

DATASETS = ["ImageNette", "CIFAR10", "CIFAR100", "STL10"]
PATH = "Images"

def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log2(np.abs(matrix).max()))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()


def format_graphs(path): 
    if not os.path.exists(path):
        return 
    f = open(path, "r")
    datasets = DATASETS
    objects = json.load(f)

    graphs = {}
    for dataset in datasets: 
        for key in objects.keys(): # key is a T#
            obj = objects[key] #this is the T object
            if graphs.get(dataset) is None: 
                graphs[dataset] = []
            if obj.get("eval_2") is not None: # if there is eval_2 it means that there is eval_1 and R1 but not R2
                if obj.get("eval_1") is not None and obj["eval_1"]["dataset_sup"]["name"] == dataset: 
                    runs = {} # here we store all the runs in each T, and we place them inside a runs object, 
                    for run in list(obj.keys()):
                        
                        new_obj = {}
                        objS = obj[run]
                        if isinstance(objS, str):
                            continue
                        new_obj["test_loss"] = objS["test_loss"]
                        new_obj["test_acc"] = objS["test_acc"]
                        new_obj["dataset"] = objS["dataset_sup"]["name"]
                        
                        if objS["dataset_sup"].get("selected_classes") is not None:
                            new_obj["selected_classes"] = objS["dataset_sup"]["selected_classes"]
                            new_obj["n_classes"] = objS["dataset_sup"]["n_classes"]
                       
                        if obj.get("model_name") is not None:
                            new_obj["model_name"] = obj["model_name"]
                        new_obj["T"] = key
                        runs[run] = new_obj# each run in runs will be of the form "R1": {fields}
                    graphs[dataset].append(runs)
            elif obj.get("R2") is not None:
                if obj.get("R1") is not None and obj["R1"]["dataset_sup"]["name"] == dataset: 
                    runs = {} # here we store all the runs in each T, and we place them inside a runs object, 
                    for run in list(obj.keys()):
                    
                        new_obj = {}
                        objS = obj[run]
                        if isinstance(objS, str):
                            continue
                        print(run)
                        if run != "cl_hyper":
                            new_obj["test_loss"] = objS["test_loss"]
                            new_obj["test_acc"] = objS["test_acc"]
                            new_obj["dataset"] = objS["dataset_sup"]["name"]
                            
                            if objS["dataset_sup"].get("selected_classes") is not None:
                                new_obj["selected_classes"] = objS["dataset_sup"]["selected_classes"]
                                new_obj["n_classes"] = objS["dataset_sup"]["n_classes"]
                            if obj.get("model_name") is not None:
                                new_obj["model_name"] = obj["model_name"]
                            new_obj["T"] = key
                        else: 
                            new_obj["T"] = key
                            new_obj["cl_hyper"] = objS
                        runs[run] = new_obj# each run in runs will be of the form "R1": {fields}
                    graphs[dataset].append(runs)
    return graphs

def create_graph(graphs, path):

    classes_CL = False
    for dataset in graphs.keys():
        if len(graphs[dataset]) > 0: 
            first_obj = (graphs[dataset])[0]
            first_run = list(first_obj.keys())[0]
            run_obj = first_obj[first_run]
            classes_CL = run_obj.get("n_classes") is not None
            break

    if classes_CL: 
        datasets = list(graphs.keys())
        for d in datasets: 
            objects = graphs[d] # it is a list of dictionaries
            for g in objects: 
                runs = list(g.keys())
                y = []
                x = runs[:-1]
                
                for run in runs: 
                    if run != "cl_hyper":
                        y.append(g[run]["test_acc"])
                        T = g[run]["T"]
                fig, ax = plt.subplots(figsize=(6, 6))  # Main plot
                run = runs[-2]

                fig.suptitle("Continual Learning with " + str(g[run]["n_classes"]) + " classes per task " + "("+ T +")")

                # Pretty-print JSON
                cl_info = json.dumps(g["cl_hyper"], indent=4)

                # Create an extra subplot for JSON text outside the main plot
                json_ax = fig.add_axes([1, 0.2, 0.2, 0.6])  # [left, bottom, width, height]
                json_ax.axis("off")  # Hide axis
                json_ax.text(0, 1, cl_info, fontsize=8, verticalalignment='top', family='monospace')

                # Plot the bar chart
                ax.bar(x, y)

                # Title and Labels
                img_name = g[run]["dataset"] + "_" + str(g[run]["n_classes"]) + "C" + "_" +T +".png"
                if g[run].get("model_name"):
                    img_name = g[run]["model_name"] +"_" +T +".png"
                    ax.set_title(g[run]["dataset"] + " on model " + g[run]["model_name"])
                else: 
                    ax.set_title(g[run]["dataset"])

                ax.set_yticks(np.arange(0, 105, 5))

                # Save and Close
                plt.savefig(path + "/" + img_name, bbox_inches='tight')
                plt.close()
                

    elif classes_CL == False:
        datasets = list(graphs.keys())
        for d in datasets: 
            objects = graphs[d] # it is a list of dictionaries
            for g in objects: 
                x = list(g.keys())
                y = []
                runs = x
                d_labels = []
                for run in runs: 
                    if run != "model_name" and run != "cl_hyper":
                        y.append(g[run]["test_acc"])
                        T = g[run]["T"]

                        if g[run]["dataset"] not in d_labels:
                            d_labels.append(g[run]["dataset"])
                plt.figure(figsize=(5, 6))
                d_title = str(d_labels[0]) + "_" + str(d_labels[1])
                plt.suptitle("Continual Learning with " + d_title + " (" + T + ")")

                # Bar plot
                plt.bar(x[:-1], y)

                # Generate image name
                img_name = d_title + "_"+ T + ".png"
                if g[run].get("model_name"):
                    img_name = g[run]["model_name"] + "_"+ T +".png"
                    plt.title("Model " + g[run]["model_name"])

                plt.yticks(np.arange(0, 105, 5))

                # Pretty-print JSON
                cl_info = json.dumps(g["cl_hyper"], indent=4)

                # Add external JSON box
                json_ax = plt.gcf().add_axes([0.9, 0.2, 0.2, 0.6])  # Moved further right (left = 0.9)
                json_ax.axis("off")  # Hide axis
                json_ax.text(0, 1, cl_info, fontsize=8, verticalalignment='top', family='monospace')

                # Save and close
                plt.savefig(path + "/" + img_name, bbox_inches='tight')
                plt.close()



if not os.path.exists(PATH): 
    os.makedirs(PATH) 

f1 = "MULTD_CL.json"
f2 = "TASKS_CL.json"

path_1 = PATH + "/" + f1[:len(f1) - 5]
path_2 = PATH + "/" + f2[:len(f2) - 5]

if not os.path.exists(path_1): 
    os.makedirs(path_1) 

if not os.path.exists(path_2): 
    os.makedirs(path_2) 


graphs_1 = format_graphs(f1)
graphs_2 = format_graphs(f2)

create_graph(graphs_1, path_1)
create_graph(graphs_2, path_2)
