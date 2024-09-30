import matplotlib.pyplot as plt
import numpy as np
import os 

import json

DATASETS = ["ImageNette", "CIFAR10", "CIFAR100", "STL10"]
PATH = "Images"

def format_graphs(path): 

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
    return graphs

def create_graph(graphs, path):

    dataset = list(graphs.keys())[0]
    first_obj = (graphs[dataset])[0]
    first_run = list(first_obj.keys())[0]
    run_obj = first_obj[first_run]
    classes_CL = run_obj.get("n_classes") is not None

    if classes_CL: 
        datasets = list(graphs.keys())
        for d in datasets: 
            objects = graphs[d] # it is a list of dictionaries
            for g in objects: 
                x = list(g.keys())
                y = []
                runs = x
                for run in runs: 
                    y.append(g[run]["test_acc"])
                plt.figure(figsize=(5, 6))
                plt.suptitle("Continual Learning with " + str(g[run]["n_classes"]) + " classes per task " + "("+ g[run]["T"]+")")
                plt.bar(x, y)
                img_name = g[run]["dataset"] + "_" + str(g[run]["n_classes"]) + "C" + ".png"
                if g[run].get("model_name"):
                    img_name = g[run]["model_name"] + ".png"
                    plt.title(g[run]["dataset"] + " on model " + g[run]["model_name"])
                else: 
                    
                    plt.title(g[run]["dataset"])

                plt.yticks(np.arange(0,105,5))

                #plt.show()
                plt.savefig(path + "/" + img_name)
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
                    y.append(g[run]["test_acc"])
                    if g[run]["dataset"] not in d_labels:
                        d_labels.append(g[run]["dataset"])
                
                plt.figure(figsize=(5, 6))
                d_title = str(d_labels[0]) + "_" + str(d_labels[1])
                plt.suptitle("Continual Learning with " + d_title + "("+ g[run]["T"]+")")
                plt.bar(x, y)
                img_name = d_title + ".png"
                if g[run].get("model_name"):
                    img_name =  g[run]["model_name"] + ".png"
                    plt.title("Model " + g[run]["model_name"])

                plt.yticks(np.arange(0,105,5))
                #plt.show()
                plt.savefig(path + "/" + img_name)
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
