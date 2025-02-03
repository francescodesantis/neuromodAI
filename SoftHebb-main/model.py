import pickle
import torch
import torch.nn as nn

try:
    from utils import RESULT, activation
except:
    from hebb.utils import RESULT, activation
from layer import generate_block
import os
import os.path as op
from engine_cl import evaluate_sup
from dataset import make_data_loaders
from utils import get_device
import numpy




def load_layers(params, model_name, resume=None, verbose=True, model_path_override=None, dataset_sup_config=None, batch_size=None, cl_hyper={}):
    """
    Create Model and load state if resume
    """
    
    if resume is not None:
        if model_path_override is None:
            model_path = op.join(RESULT, 'network', model_name, 'models', 'checkpoint.pth.tar')
        else:
            model_path = model_path_override

        if op.isfile(model_path):
            checkpoint = torch.load(model_path)  # , map_location=device)
            state_dict = checkpoint['state_dict']
            params2 = checkpoint['config']
            if resume == 'without_classifier':
                classifier_key = list(params.keys())[-1]
                params2[classifier_key] = params[classifier_key]

            print("PARAMSSSS IN LOAD: ", params2)
            # print("state_dict.keys(): ", state_dict.keys())

            # print("state_dict: ", state_dict)
            cl_hyper["heads_basis_t"]=float(checkpoint["heads_thresh"])
            model = MultiLayer(params2, acts=checkpoint["acts"], avg_deltas=checkpoint["avg_deltas"], heads=checkpoint["heads"], cl_hyper=cl_hyper)
            


            state_dict2 = model.state_dict()

            if resume == 'without_classifier':
                for key, value in state_dict.items():
                    if resume == 'without_classifier' and str(params[classifier_key]['num']) in key:
                        continue
                    if key in state_dict2:
                        state_dict2[key] = value
                if cl_hyper["head_sol"] and dataset_sup_config is not None and batch_size is not None:
                    #call best_head
                    model.to(get_device())
                    chosen_head = best_head(model, state_dict2, dataset_sup_config, batch_size)
                    keys = list(chosen_head.keys())
                    state_dict2[keys[0]] = chosen_head[keys[0]]
                    state_dict2[keys[1]] = chosen_head[keys[1]]
                model.load_state_dict(state_dict2)
            else:
                if cl_hyper["head_sol"] and dataset_sup_config is not None and batch_size is not None:
                    #call best_head
                    model.to(get_device())
                    chosen_head = best_head(model, state_dict, dataset_sup_config, batch_size)
                    print("chosen_head: ", chosen_head)
                    keys = list(chosen_head.keys())
                    state_dict[keys[0]] = chosen_head[keys[0]]
                    state_dict[keys[1]] = chosen_head[keys[1]]
                model.load_state_dict(state_dict)
            # log.from_dict(checkpoint['measures'])
            starting_epoch = 0  # checkpoint['epoch']
            print('\n', 'Model %s loaded successfuly with best perf' % (model_name))
            # shutil.rmtree(op.join(RESULT, params.folder_name, 'figures'))
            # os.mkdir(op.join(RESULT, params.folder_name, 'figures'))
        else:
            print('\n', 'Model %s not found' % model_name)
            
            model = MultiLayer(params, cl_hyper=cl_hyper)
        print('\n')
    else:
        model = MultiLayer(params, cl_hyper=cl_hyper)

    if verbose:
        model.__str__()
    print("model.heads: ", model.heads)
    return model

def best_head(model, state_dict, dataset_sup_config, batch_size): 
# Evaluates all the heads in the model and returns the best one, if none is found, 
# meaning we don't reach the threshold for the accuracy, we return a newly initialized head.
    keys = list(state_dict.keys())
    chosen_head = { keys[-1]:state_dict[keys[-1]], keys[-2]: state_dict[keys[-2]]}
    chosen_acc = 0
    device = get_device()
    avg_acc = 0

    initial_state = state_dict
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = make_data_loaders(dataset_sup_config, batch_size, device)
    for head in model.heads: 
        keys = list(head.keys())
        if keys[0] in state_dict and keys[1]  in state_dict:
            state_dict[keys[0]] = head[keys[0]]
            state_dict[keys[1]] = head[keys[1]]
        model.load_state_dict(state_dict)
        loss_test, acc_test = evaluate_sup(model, criterion, test_loader, device)
        acc_test = acc_test/100
        avg_acc += acc_test
        print("test_acc: ", acc_test, model.heads_thresh)
        if acc_test > chosen_acc: 
            chosen_head = head
            chosen_acc = acc_test
            print("chosen_acc: ", chosen_acc)
    
    #model.heads.remove(chosen_head)
    return chosen_head



def save_layers(model, model_name, epoch, blocks, filename='checkpoint.pth.tar', storing_path=None):
    """
    Save model and each of its training blocks
    """

    if storing_path is None:
        print("STORING PATH IS NONEEEEEE")
        if not op.isdir(RESULT):
            os.makedirs(RESULT)
        if not op.isdir(op.join(RESULT, 'network')):
            os.mkdir(op.join(RESULT, 'network'))
            os.mkdir(op.join(RESULT, 'layer'))

        folder_path = op.join(RESULT, 'network', model_name)
        if not op.isdir(folder_path):
            os.makedirs(op.join(folder_path, 'models'))
        storing_path = op.join(folder_path, 'models')

    print("SAVING THE MODEL")
    print(storing_path)
    
    
    
        
    

    #print("Current stored value: ", model.acts["conv0"][:5], model.avg_deltas['blocks.0.layer.weight'][:5] )
    print("SAVED HEADS THRESHOLD: ", model.heads_thresh)
    print("SAVED model.acts", model.acts)
    print("SAVED model.heads_thresh", model.heads_thresh)
    torch.save({
        'state_dict': model.state_dict(),
        'config': model.config,
        'avg_deltas': model.avg_deltas,
        'acts': model.acts,
        'epoch': epoch, 
        'heads': model.heads, 
        'heads_thresh' : model.heads_thresh, 
    }, op.join(storing_path, filename))

    for block_id in blocks:
        block = model.get_block(block_id)
        block_path = op.join(RESULT, 'layer', 'block%s' % block.num)
        if not op.isdir(block_path):
            os.makedirs(block_path)
        folder_path = op.join(block_path, block.get_name())
        if not op.isdir(folder_path):
            os.mkdir(folder_path)
        torch.save({
            'state_dict': block.state_dict(),
            'epoch': epoch
        }, op.join(folder_path, filename))






class MultiLayer(nn.Module):
    """
       MultiLayer Network created from list of preset blocks
    """

    def __init__(self, blocks_params: dict, blocks: nn.Module = None, acts: dict = {}, avg_deltas: dict = {}, heads: list = [], cl_hyper: dict = {}) -> None:
        super().__init__()
        self.train_mode = None
        self.train_blocks = []
        self.storage = 0
        ########################################################
        file_path_d = 'avg_deltas.p'
        file_path_act = 'activations.p'
        
        self.avg_deltas = avg_deltas
        self.acts = acts
        self.heads = heads
        self.heads_thresh = cl_hyper["heads_basis_t"]
        self.cl_hyper = cl_hyper
        
        # if os.path.exists(file_path_d):
        #     with open('avg_deltas.p', 'rb') as pfile:
        #         avg_deltas = pickle.load(pfile)
        
        # if os.path.exists(file_path_act):
        #     with open('activations.p', 'rb') as pfile:
        #         acts = pickle.load(pfile)
        ########################################################

        self.config = blocks_params
        layer_num = 0
        avg_deltas_layer = None
        acts_layer = None
        depth = len(blocks_params)-1
        if avg_deltas is not None and acts_layer is not None: 
            print(avg_deltas.keys())
            print(acts_layer.keys())
        else: 
            print("avg_deltas: ", avg_deltas)
            print("acts_layer: ", acts_layer)
        if blocks_params is not None:
            blocks = []
            for _, params in blocks_params.items():
                # params : {'arch': 'CNN', 'preset': 'softkrotov-c1536-k3-p1-s1-d1-b0-t0.25-lr0.01-lp0.5-e0', 'operation': 'batchnorm2d', 'num': 2, 'batch_norm': False, 'pool': {'type': 'avg', 'kernel_size': 2, 'stride': 2, 'padding': 0}, 'activation': {'function': 'triangle', 'param': 1.0}, 'resume': None, 'layer': {'arch': 'CNN', 'nb_train': None, 'lr': 0.01, 'adaptive': True, 'lr_sup': 0.001, 'speed': 7, 'lr_div': 96, 'lebesgue_p': 2, 'padding_mode': 'reflect', 'pre_triangle': False, 'ranking_param': 3, 'delta': 2, 't_invert': 0.25, 'groups': 1, 'stride': 1, 'dilation': 1, 'beta': 1, 'power': 4.5, 'padding': 1, 'weight_init': 'normal', 'weight_init_range': 0.4252586358998573, 'weight_init_offset': 0, 'mask_thsd': 0, 'radius': 25, 'power_lr': 0.5, 'weight_decay': 0, 'soft_activation_fn': 'exp', 'hebbian': True, 'resume': None, 'add_bias': False, 'normalize_inp': False, 'lr_decay': 'linear', 'seed': 0, 'softness': 'softkrotov', 'out_channels': 1536, 'kernel_size': 3, 'in_channels': 384, 'lr_scheduler': {'lr': 0.01, 'adaptive': True, 'nb_epochs': 1, 'ratio': 0.0002, 'speed': 7, 'div': 96, 'decay': 'linear', 'power_lr': 0.5}}}
                if params['arch'] == 'CNN':
                    if len(self.avg_deltas) > 0:
                        print("self.avg_deltas inside model: ", self.avg_deltas) 
                        avg_deltas_layer = self.avg_deltas["blocks." +  str(layer_num) + ".layer.weight"]
                        print('"blocks." +  str(layer_num) + ".layer.weight": ', "blocks." +  str(layer_num) + ".layer.weight") 

                        print("self.avg_deltas_layer inside model after retrieval: ", avg_deltas_layer) 
                    if len(self.acts) > 0:
                        print("self.acts inside model: ", self.avg_deltas)
                        if layer_num == depth: 
                            acts_layer = self.acts["linear" + str(layer_num)]
                        else:
                            acts_layer = self.acts["conv" + str(layer_num)]
                        print("acts_layer inside model after retrieval: ", acts_layer) 
                blocks.append(generate_block(params, avg_deltas_layer, acts_layer, cl_hyper))
                layer_num += 1
            self.blocks = nn.Sequential(*blocks)
        else:
            self.blocks = nn.Sequential(*blocks)

    def foward_x_wta(self, x: torch.Tensor) -> torch.Tensor:
        for id, block in self.generator_block():  
            if id != len(self.blocks) - 1:
                x = block(x)   # block is an instance of a layer or a more complex sequence of layers (often called a block or a module). 
                                # so we apply the transformation to the input x for all the components in block 
                                # and then the resulting output will be passed forward to the 
                                # to the next block until we have done them all. 
            else:
                return block.foward_x_wta(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        return x

    def get_block(self, id):
        return self.blocks[id]

    def sub_model(self, block_ids):
        sub_blocks = []
        max_id = max(block_ids)
        for id, block in self.generator_block():
            sub_blocks.append(self.get_block(id))
            if id == max_id:
                break

        return MultiLayer(None, sub_blocks)

    def is_hebbian(self) -> bool:
        """
        Return if the last block of the model is hebbian
        """
        return self.blocks[-1].is_hebbian()

    def get_lr(self) -> float:
        """
        Return the lr of the last hebbian block
        """
        if self.train_blocks:
            for i in reversed(self.train_blocks):
                if self.blocks[-i].is_hebbian():
                    return self.blocks[-i].get_lr()
        if self.blocks[0].is_hebbian():
            return self.blocks[0].get_lr()
        return 0

    def radius(self, layer=None) -> str:
        """
        Return the radius of the first hebbian block
        """
        if layer is not None:
            return self.blocks[layer].radius()
        if self.train_blocks:
            r = []
            for i in reversed(self.train_blocks):
                if self.blocks[i].is_hebbian():
                    r.append(self.blocks[i].radius())
            return '\n ************************************************************** \n'.join(r)
        if self.blocks[0].is_hebbian():
            return self.blocks[0].radius()
        return ''

    def convergence(self) -> str:
        """
        Return the radius of the last hebbian block
        """
        for i in range(1, len(self.blocks) + 1):
            if self.blocks[-i].is_hebbian():
                return self.blocks[-i].layer.convergence()
        return 0, 0

    def reset(self):
        if self.blocks[0].is_hebbian():
            self.blocks[0].layer.reset()

    def generator_block(self):
        for id, block in enumerate(self.blocks):
            yield id, block

    def update(self):
        for block in self.train_blocks:
            self.get_block(block).update()

    def __str__(self):
        for _, block in self.generator_block():
            block.__str__()

    def train(self, mode=True, blocks=[]):
        """
        Set the learning update to the expected mode.
        mode:True, BP:False, HB:True --> training Hebbian layer
        mode:True, BP:True, HB:False --> training fc
        mode:True, BP:True, HB:True --> training Hebbain + fc blocks
        mode:False --> predict
        """
        self.training = mode
        self.train_blocks = blocks
        # print('train mode %s and layer %s'%(mode, blocks))

        for param in self.parameters():
            param.requires_grad = False
        for _, block in self.generator_block():
            block.eval()

        for block in blocks:
            module = self.get_block(block)

            module.train(mode)
            for param in module.parameters():
                param.requires_grad = True


class HebbianOptimizer:
    def __init__(self, model):
        """Custom optimizer which particularly delegates weight updates of Unsupervised layers to these layers themselves.

        Args:
            model (torch.nn.Module): Pytorch model
        """
        self.model = model
        self.param_groups = []

    @torch.no_grad()
    def step(self, *args):
        """Performs a single optimization step.
        """
        loss = None

        for block in self.model.blocks:
            if block.is_hebbian():
                block.update(*args)

    def zero_grad(self):
        pass


class AggregateOptim:
    def __init__(self, optimizers):
        """Custom optimizer aggregating several optimizers together to run simulaneously

        Args:
            optimizers (List[torch.autograd.optim.Optimizer]): List of optimizers which need to be called simultaneously
        """
        self.optimizers = optimizers
        self.param_groups = []
        for optim in self.optimizers:
            self.param_groups.extend(optim.param_groups)

    def __repr__(self):
        representations = []
        for optim in self.optimizers:
            representations.append(repr(optim))
        return '\n'.join(representations)

    def step(self):
        for optim in self.optimizers:
            optim.step()

    def zero_grad(self):
        for optim in self.optimizers:
            optim.zero_grad()
