import torch
import torch.nn as nn
import time
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train_BP(model, criterion, optimizer, loader, device, measures):
    """
    Train only the traditional blocks with backprop
    """
    # with torch.autograd.set_detect_anomaly(True):
    t = time.time()
    for inputs, target in loader:
        ## 1. forward propagation$
        inputs = inputs.float().to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(inputs)
        # print(r"%s" % (time.time() - t))

        ## 2. loss calculation
        loss = criterion(output, target)

        ## 3. compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(optimizer.param_groups)

        ## 4. Accuracy assessment
        predict = output.data.max(1)[1]

        acc = predict.eq(target.data).sum()
        # Save if measurement is wanted

        # print(model.blocks[1].layer.weight.mean(), model.blocks[1].layer.weight.std())

        convergence, R1 = model.convergence()
        measures.step(target.shape[0], loss.clone().detach().cpu(), acc.cpu(), convergence, R1, model.get_lr())

    return measures, optimizer.param_groups[0]['lr']

"""
The first thing we do is check if the model is hebbian or not (basically if it is we set the loss accuracy to False).
Then we tell torch not to calculate any gradient because we don't need any for unsupervised hebbian. 
We don't get inside the if loss_acc clause why???
model.is_hebbian() returns true if the last block of the model is hebbian or not and checks if the criterion is not none.
The criterion can be something like ... ??? criterion seems to be none always, just like measures. 
So are they both to be defined??? 


"""

def train_hebb(model, loader, device, measures=None, criterion=None):
    """
    Train only the hebbian blocks
    """
    t = time.time()
    #print("LOADER VARIABLE")
    #print(loader)
    #print("#############################################")
    loss_acc = (not model.is_hebbian()) and (criterion is not None)
    t = True
    with torch.no_grad(): #Context-manager that disables gradient calculation.
        for inputs, target in loader:

            # print(inputs.min(), inputs.max(), inputs.mean(), inputs.std())
            ## 1. forward propagation
            inputs = inputs.float().to(device)  # , non_blocking=True) send the data to the device (GPU)
            output = model(inputs) 
            if t == False:
                # print("INPUT VARIABLE")
                # print(inputs)
                # print("#############################################")
                # print("OUTPUT VARIABLE")
                # print(output)
                # print("#############################################")
                t = True
            # print(r"%s"%(time.time()-t))

            if loss_acc:  
                target = target.to(device, non_blocking=True)
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! INSIDE LOSS ACCURACY")
                
                print("#############################################")
                ## 2. loss calculation
                loss = criterion(output, target)   

                ## 3. Accuracy assessment
                predict = output.data.max(1)[1]
                acc = predict.eq(target.data).sum()
                # Save if measurement is wanted
                conv, r1 = model.convergence()
                measures.step(target.shape[0], loss.clone().detach().cpu(), acc.cpu(), conv, r1, model.get_lr())
            print("STATE DICT: ###############################")
            for param_tensor in model.state_dict():
                print(param_tensor, "\t", model.state_dict()[param_tensor].size(), model.state_dict()[param_tensor])

            model.update()

    info = model.radius()
    convergence, R1 = model.convergence()
    return measures, model.get_lr(), info, convergence, R1

""" 
Again we set the gradient to zero.
There is still the problem with the criterion which looks like to be always none and not defined anywhere.
We then load the input to the device and calculate the output.

The model.blocks.plasticity function allow to calculate the weight change vector which is done only the last layer. 
why??
Let's reason a little bit, we have to update the weights, now this could either be something which dpenedds on the type of
ùoperations that we are performing or it could be something which derives from the nature of the input itself. Could it be that 
since we are working with networks which have at most one hebbian layer??? I don't understand, this need to be investigated 
a bit more. 

Here we don't have the same issue we have for the unsupervised learning where we never get into the if loss_acc clause 
because the criterion and the measures are never passed. In this case the criterion passed is  criterion = nn.CrossEntropyLoss()
and the measures is   log_batch = log.new_log_batch() which is defined in run_sup() and needs to be investigated.
There is the possibility of entering the clause but I don't understand how we manage to do it: 
the model is supposed to be not hebbian to calculate the loss... right, but if all the models we 
consider are hebbian then what? this is train sup hebbian function... wth it doesn't make sense. 
Let's try and analyze it: 
to enter the if clause we need to not be hebbian, which is set through the flag is_hebbian, which is set to true
if  when we read the preset flag in the object contained in the preset.json we read anyhing but MLP. Then we need
to have a criterion which is not none and we aldready saw that when we call it the object passed is the 
cross entropy loss criteria. So we just need to understand when the flag for hebbian is set to false. Like what is 
the role of this function because from my initial understanding it was to train the hebbian learning in a 
supervised manner, which kindd of doesn't make sense because if we are working with an hebbian network where
should we utilize the feedback given from the supervised approach? This is used only in the case the model is not hebbian, 
So now the question becomes when is the model not hebbian??
The is_hebbian function returns true by checking only the last block. If the last block has the hebbian flag set to true
then is_hebbian return  true, then when is the last layer set to hebbian? We check if the preset is field is either soft or BP. 
If it is BP we set the hebbian flag to false otherwise we set it to true. 

"2SoftMlpMNIST": {
      "b0": {
        "arch": "MLP",
        "preset": "soft-c2000-t12-lr0.045-r35-v1",
        "operation": "flatten",
        "activation": "softmax_5",
        "num": 0,
        "batch_norm": false
      },
      "b1": {
        "arch": "MLP",
        "operation": "",
        "preset": "BP-c10",      ------------------------------------> we check this field here 
        "dropout": 0,
        "num": 1
      }

      Ok so we got how the whole thing works, but then why are we just using 1 block?? 
      By this I mean that if the check is done on the last block only and this block is hebbian because we dont get in the
      if then we must be using only one block not considerign the second one which is always using back prop for classification.

    UPDATE: when we call this train_sup_hebb the is_hebbian is set to false... almost looks like it splits the model down in
    two parts when we have to train the hebbian part we call the run_unsup and when train the classificator we call run_sup, 
    which is why the length of blocks is just one... the classificator block is always one! Ok so the division is done in ray search when we
    check in the config if the mode is unsupervised or supervised. 



"""
def train_sup_hebb(model, loader, device, measures=None, criterion=None):
    """
    Train only the hebbian blocks

    """
    t = time.time()
    loss_acc = (not model.is_hebbian()) and (criterion is not None)
    print("LOSS_ACC: ", loss_acc )
    with torch.no_grad():
        for inputs, target in loader:
            # print(inputs.min(), inputs.max(), inputs.mean(), inputs.std())
            ## 1. forward propagation
            inputs = inputs.float().to(device)
            output = model(inputs)
            model.blocks[-1].layer.plasticity(x=model.blocks[-1].layer.forward_store['x'],
                                              pre_x=model.blocks[-1].layer.forward_store['pre_x'],
                                              wta=torch.nn.functional.one_hot(target, num_classes=
                                              model.blocks[-1].layer.forward_store['pre_x'].shape[1]).type(
                                              model.blocks[-1].layer.forward_store['pre_x'].type()))

            if loss_acc:

                print("INSIDE LOSS_ACC OF train_sup_hebb")
                target = target.to(device, non_blocking=True)

                ## 2. loss calculation
                loss = criterion(output, target)

                ## 3. Accuracy assessment
                predict = output.data.max(1)[1]
                acc = predict.eq(target.data).sum()
                # Save if measurement is wanted
                conv, r1 = model.convergence()
                measures.step(target.shape[0], loss.clone().detach().cpu(), acc.cpu(), conv, r1, model.get_lr())

            model.update()

    info = model.radius()
    convergence, R1 = model.convergence()
    return measures, model.get_lr(), info, convergence, R1

""""""
def train_unsup(model, loader, device,
                blocks=[]):  # fixed bug as optimizer is not used or pass in the only use it has in this repo currently
    """
    Unsupervised learning only works with hebbian learning
    """
    model.train(blocks=blocks)  # set unsup blocks to train mode
    _, lr, info, convergence, R1 = train_hebb(model, loader, device)
    return lr, info, convergence, R1

"""
This function performs the training of the supervised learning part of the model.
The first thing we do is check if the number of blocks is = 1, but why??? 
Then we check if the first block is hebbian, if so we use train_sup_hebb().
otherwise it can be hybrid ( which implies tht there are more than just one block ) or simply the classical Back Prop.
"""
def train_sup(model, criterion, optimizer, loader, device, measures, learning_mode, blocks=[]):
    """
    train hybrid model.
    learning_mode=HB --> train_hebb
    learning_mode=BP --> train_BP
    """
    if len(blocks) == 1:
        model.train(blocks=blocks)
        #print(model.is_hebbian())
        if model.get_block(blocks[0]).is_hebbian():
            measures, lr, info, convergence, R1 = train_sup_hebb(model, loader, device, measures, criterion)
        else:
            measures, lr = train_BP(model, criterion, optimizer, loader, device, measures)
    else:
        model.train(blocks=blocks)
        if learning_mode == 'HB':
            measures, lr, info, convergence, R1 = train_sup_hebb(model, loader, device, measures, criterion)
        else:
            measures, lr = train_BP(model, criterion, optimizer, loader, device, measures)
    return measures, lr


def evaluate_unsup(model, train_loader, test_loader, device, blocks):
    """
    Unsupervised evaluation, only support MLP architecture

    """
    print("INSIDE EVALUATE UNSUP")
    #print(blocks)
    if model.get_block(blocks[-1]).arch == 'MLP':
        sub_model = model.sub_model(blocks)
        return evaluate_hebb(sub_model, train_loader, test_loader, device)
    else:
        print("INSIDE EVALUATE UNSUP RETURNED 0,0")

        return 0., 0.


def evaluate_hebb(model, train_loader, test_loader, device):
    if train_loader.dataset.split == 'unlabeled':
        print('Unalbeled dataset, cant perform unsupervised evaluation')
        return 0, 0
    print("INSIDE EVALUATE HEBB")

    preactivations, winner_ids, neuron_labels, targets = infer_dataset(model, train_loader, device)
    acc_train = get_accuracy(model, winner_ids, targets, preactivations, neuron_labels, device)

    preactivations_test, winner_ids_test, _, targets_test = infer_dataset(model, test_loader, device)
    acc_test = get_accuracy(model, winner_ids_test, targets_test, preactivations_test, neuron_labels, device)
    return float(acc_train.cpu()), float(acc_test.cpu())


"""
we take the model and call eval to turn off batch normalization layers, dropout and so on 
because we need to put the model in inference mode. 
We then take all the labels ( targets ).
Then we load all the input to the gpu setting the non blocking flag to true: 
the non_blocking flag is used in data transfer operations between CPU and GPU memory. 
When this flag is set to True, it allows the transfer to be asynchronous, meaning it does not 
block the execution of the program while waiting for the data transfer to complete.

After that we take the preactivations and the wta from the forward_x_wta
wta: are th
"""
def infer_dataset(model, loader, device):
    model.eval()
    targets_lst = []
    winner_ids = []
    preactivations_lst = []
    print("INSIDE INFER DATA")

    wta_lst = []
    with torch.no_grad():
        for inputs, targets in loader:
            ## 1. forward propagation
            inputs = inputs[targets != -1]
            targets = targets[targets != -1]
            if targets.nelement() != 0:
                inputs = inputs.float().to(device, non_blocking=True)
                preactivations, wta = model.foward_x_wta(inputs)
                print("WTA: ", wta)
                print("PREACTIVATIONS: ", preactivations)
                preactivations_lst.append(preactivations)
                wta_lst.append(wta)
                targets_lst += targets.tolist()
                winner_ids_minibatch = wta.argmax(dim=1)
                winner_ids += winner_ids_minibatch.tolist()

    winner_ids = torch.FloatTensor(winner_ids).to(torch.int64).to(device)
    targets = torch.FloatTensor(targets_lst).to(torch.int64).to(device)
    preactivations = torch.cat(preactivations_lst).to(device)
    wta = torch.cat(wta_lst).to(device)
    neuron_labels = get_neuron_labels(model, winner_ids, targets, preactivations, wta)
    return preactivations, winner_ids, neuron_labels, targets


def evaluate_sup(model, criterion, loader, device):
    """
    Evaluate the model, returning loss and acc
    """
    model.eval()
    loss_sum = 0
    acc_sum = 0
    n_inputs = 0

    with torch.no_grad():
        for inputs, target in loader:
            ## 1. forward propagation
            inputs = inputs.float().to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(inputs)

            ## 2. loss calculation
            loss = criterion(output, target)
            loss_sum += loss.clone().detach()

            ## 3. Accuracy assesment
            predict = output.data.max(1)[1]
            acc = predict.eq(target.data).sum()
            acc_sum += acc
            n_inputs += target.shape[0]

    return loss_sum.cpu() / n_inputs, 100 * acc_sum.cpu() / n_inputs


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


"""Code needs to be rewrite"""


def get_neuron_labels(model, winner_ids, targets, preactivations, wta):
    targets_onehot = nn.functional.one_hot(targets, num_classes=preactivations.shape[1]).to(torch.float32)
    winner_ids_onehot = nn.functional.one_hot(winner_ids, num_classes=preactivations.shape[1]).to(torch.float32)
    responses_matrix = torch.matmul(winner_ids_onehot.t(), targets_onehot)

    neuron_outputs_for_label_total = torch.matmul(wta.t(), targets_onehot)

    responses_matrix[responses_matrix.sum(dim=1) == 0] = neuron_outputs_for_label_total[
        responses_matrix.sum(dim=1) == 0]
    neuron_labels = responses_matrix.argmax(1)
    return neuron_labels


def get_accuracy(model, winner_ids, targets, preactivations, neuron_labels, device):
    n_samples = preactivations.shape[0]
    # if not model.ensemble:
    predlabels = torch.FloatTensor([neuron_labels[i] for i in winner_ids]).to(device)
    '''
    else:
        if model.test_uses_softmax:
            soft_acts = activation(preactivations, model.t_invert, model.activation_fn, dim=1, power=model.power, normalize=True)
            winner_ensembles = [
                np.argmax([np.sum(np.where(neuron_labels == ensemble, soft_acts[sample], np.asarray(0))) for
                           ensemble in range(10)]) for sample in range(n_samples)]
        else:
            winner_ensembles = [
                np.argmax([np.sum(np.where(neuron_labels == ensemble, preactivations[sample], np.asarray(0))) for
                           ensemble in range(10)]) for sample in range(n_samples)]
        predlabels = winner_ensembles
    '''
    correct_pred = predlabels == targets
    n_correct = correct_pred.sum()
    accuracy = n_correct / len(targets)
    return 100 * accuracy.cpu()
