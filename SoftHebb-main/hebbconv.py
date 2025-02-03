import torch
import torch.nn as nn
from typing import Generator, Union
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from utils import DEVICE
try:
    from utils import init_weight, activation, unsup_lr_scheduler
except:
    from hebb.utils import init_weight, activation, unsup_lr_scheduler
import einops
from tabulate import tabulate

from activation import Triangle
global xyz
import numpy as np
R = [True]
def myprint(t):
    if R[0]: 
        print("WTA IN delta_weight: ", t)
        R[0] = False

class HebbHardConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            lebesgue_p: int,
            kernel_size: int,
            weight_distribution: str,
            weight_range: float,
            pre_triangle: bool,
            mask_thsd: float,
            lr_scheduler: str,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            padding_mode: str = "zeros",
            bias: bool = False,
            nb_train: int = None,
            # avg_deltas_layer: torch.tensor = None,
            # top_acts_layer: list = None

    ) -> None:
        """
        Hard Winner take all convolutional layer implementation
        """

        super(HebbHardConv2d, self).__init__()
        self.temp = 0
        self.learning_update = False
        self.was_update = True
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding_mode = padding_mode
        self.dilation = _pair(dilation)
        self.padding_mode = padding_mode
        self.groups = groups
        self.out_channels_groups = out_channels // groups
        self.lebesgue_p = lebesgue_p

        self.add_threshold = True
        self.pre_triangle = pre_triangle
        if pre_triangle:
            print('preactivation using triangle')
            self.triangle = Triangle(1)

        self.stat = torch.zeros(3, out_channels)
        self.F_pad = padding_mode != 'zeros'
        if self.F_pad:
            self.padding = _pair(0)
            self.F_padding = (padding, padding, padding, padding)
        else:
            self.padding = _pair(padding)

        '''
        if self.add_threshold:
            self.register_buffer("threshold", torch.zeros((1,out_channels,1,1)), persistent=False)
            self.target_rate = 1/out_channels
        '''

        self.register_buffer(
            'weight',
            init_weight(
                (out_channels, in_channels // groups, *self.kernel_size),
                weight_distribution=weight_distribution,
                weight_range=weight_range
            )
        )
        '''
                self.register_buffer(
            'mask',
            (init_weight(
                (out_channels, in_channels // groups, *self.kernel_size),
                weight_distribution='positive',
                weight_range=1
            ) > mask_thsd).float(),
            persistent=False
        )
                self.register_buffer("delta_w", torch.zeros_like(self.weight), persistent=False)
                self.weight = self.weight * self.mask
        '''
        self.register_buffer("delta_w", torch.zeros_like(self.weight), persistent=False)

        self.register_buffer(
            "rad",
            torch.ones(out_channels),
            persistent=False
        )
        self.get_radius()

        self.mask_thsd = mask_thsd

        self.initial_weight = self.weight.clone()

        self.nb_train = self.out_channels if nb_train is None else nb_train

        self.lr_scheduler_config = lr_scheduler.copy()
        self.lr_adaptive = self.lr_scheduler_config['adaptive']

        self.reset()

        self.conv = 0

        self.register_buffer('bias', None)

        
        

        

    def reset(self):
        if self.lr_adaptive:
            self.register_buffer("lr", torch.ones_like(self.weight), persistent=False)
            self.lr_scheduler = unsup_lr_scheduler(lr=self.lr_scheduler_config['lr'],
                                                   nb_epochs=self.lr_scheduler_config['nb_epochs'],
                                                   ratio=self.lr_scheduler_config['ratio'],
                                                   speed=self.lr_scheduler_config['speed'],
                                                   div=self.lr_scheduler_config['div'],
                                                   decay=self.lr_scheduler_config['decay'])

            self.update_lr()
        else:

            self.lr_scheduler = unsup_lr_scheduler(lr=self.lr_scheduler_config['lr'],
                                                   nb_epochs=self.lr_scheduler_config['nb_epochs'],
                                                   ratio=self.lr_scheduler_config['ratio'],
                                                   speed=self.lr_scheduler_config['speed'],
                                                   div=self.lr_scheduler_config['div'],
                                                   decay=self.lr_scheduler_config['decay'])
            self.lr = next(self.lr_scheduler)

    def reset_weight(self):
        weight = self.weight.view(self.weight.shape[0], -1)

        norm = torch.linalg.norm(weight, dim=1, ord=self.lebesgue_p)
        # mean_radius = torch.mean(norm)
        # mean2_radius = torch.mean(torch.abs(norm - mean_radius * torch.ones_like(norm)))

        no_r1 = torch.abs(norm - torch.ones_like(norm)) > 5e-3
        self.weight[no_r1] = self.initial_weight[no_r1].to(self.weight.device)

    def get_pre_activations(self, x: torch.Tensor, mask: bool = False) -> torch.Tensor:
        """
        Compute the preactivation or the current of the hebbian layer
        ----------
        x : torch.Tensor
            Input
        x : torch.Tensor
            Pre activation
        Returns
        -------
        pre_x : torch.Tensor
            Current of the hebbian layer
        """
        # the calulation of the preactivations is done through the convolution operation of the input x 
        # and the weights associated to the current layer. Store the signs of the weight vector in a tensor, 
        # so that we can consider the absolute value of the tensor of the weights to the power of (self.lebesgue_p - 1)
        # and then restore the signs of the single weights by multiplying it with torch.sign(self.weight).
        
        pre_x = self._conv_forward(
            x,
            torch.sign(self.weight) * torch.abs(self.weight) ** (self.lebesgue_p - 1),
            self.bias
        )
        return pre_x

    def get_lr(self):
        if self.lr_adaptive:
            return self.lr.mean().cpu()
        return self.lr

    def get_wta(self, pre_x: torch.Tensor, group_id: int = 0) -> torch.Tensor:
        """
        Compute the hard winner take all
        ----------
        pre_x : torch.Tensor
            Input
        Returns
        -------
        wta : torch.Tensor
            preactivation or the current of the hebbian layer

        We take the wta vector (in one hot encoding form) of the pre activations to find maximum activation neuron.

        where pre_x is of shape [batch_size, out_channels, height_out, width_out], where out_channels

        """
        # wta = nn.functional.one_hot(pre_x.argmax(dim=0), num_classes=pre_x.shape[0]).to(
        #    torch.float).permute(3,0,1,2)
        batch_size, out_channels, height_out, width_out = pre_x.shape
        pre_x_flat = pre_x.transpose(0, 1).reshape(out_channels, -1)
        #print(pre_x.shape)

        # num of classes corresponds to how many elements are in each vector, in this case we take the shape[0] of pre_x_flat,
        # which corresponds to 
        wta = nn.functional.one_hot(pre_x_flat.argmax(dim=0), num_classes=pre_x_flat.shape[0]).to(
            torch.float)
        self.stat[2, group_id * self.out_channels_groups: (group_id + 1) * self.out_channels_groups] += wta.sum(0).cpu()
        #print(torch.histc(wta.sum(1), bins=out_channels))
        wta = wta.transpose(0, 1).view(
            out_channels, batch_size, height_out, width_out
        ).transpose(0, 1)
        #print(torch.histc(wta.sum(), bins=96))
        #print("WTA[0]", wta[0])
        return wta

    def stat_wta(self):
        count = self.stat.clone()
        count[2:] = (100 * self.stat[2:].t() / self.stat[2:].sum(1)).t()
        count = count[:, :30]
        x = list(range(count.shape[1]))
        y = [['{lr:.2e}'.format(lr=lr) for lr in count[0].tolist()]] + [['{x:.2f}'.format(x=x) for x in y.tolist()] for
                                                                        y in count[1:]]
        return tabulate(y, headers=x, tablefmt='orgtbl')

    def _conv_forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        '''Convolutional forward method. Copied from pytorch repo with only padding_mode to zeros

        Parameters:
        -----------
        input : tensor,
            Input image, channels_first format.

        weight : tensor, optional,
            Convolution kernels. If not given, self.weight is used.
        '''
        return F.conv2d(x, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, x: torch.Tensor, return_x_wta: bool = False) -> torch.Tensor:
        """
        Compute output of the layer (forward pass).
        Parameters
        ----------
        x : torch.Tensor
            Input. Expected to be of shape (batch_size, ...), where ... denotes an arbitrary
            sequence of dimensions, with product equal to in_features.
        """
        if self.F_pad:
            x = F.pad(x, self.F_padding, self.padding_mode)

        pre_x = self.get_pre_activations(x, mask=True)

        if self.pre_triangle:
            pre_x = self.triangle(pre_x)

        # If propagation of preAcitvations only no need to do the rest
        if not self.learning_update and not return_x_wta:
            return pre_x

        if self.nb_train > 0:
            pre_x = torch.chunk(pre_x, self.groups, dim=1)
            # wta = [self.get_wta(pre_x_group.clone(), group_id) for group_id, pre_x_group in enumerate(pre_x)]
            wta = []
            # pre_x.shape=(batch_size,channels,height,width)
            # so we calculate the i_wta for every single batch and then append it to the list wta.
            for group_id, pre_x_group in enumerate(pre_x):
                i_wta = self.get_wta(pre_x_group.clone(), group_id)
                wta.append(i_wta)
            #print("pre_x in forward: ", pre_x.cpu().shape)
            if return_x_wta:
                wta = torch.cat(wta)
                return pre_x.reshape(pre_x.shape[0], -1), wta.reshape(wta.shape[0], -1)

            x = torch.chunk(x, self.groups, dim=1)
            if self.learning_update:
                self.plasticity(x, pre_x, wta)

        return torch.cat(pre_x, dim=1)

    def train(self, mode: bool = True) -> None:
        """
        Set the learning update to the mode expected.
        mode:True --> training

        mode:False --> predict
        """
        self.learning_update = mode
    def delta_weight(
            self,
            x: torch.Tensor,
            pre_x: torch.Tensor,
            wta: torch.Tensor,
            weight: torch.tensor
    ) -> torch.Tensor:
        # delta_weight(x_group, pre_x_group, wta_group, weight_group)
        """
        Compute the change of weights
        Parameters
        ----------
        x : torch.Tensor
            x. Input (batch_size, in_features).
        pre_x : torch.Tensor
            pre_x. Linear transformation of the input (batch_size, in_features).
        wta : torch.Tensor
            wta. Winner take all (batch_size, in_features).
        Returns
        -------
            delta_weight : torch.Tensor
        """
        # here we apply the convolution operation on the inputs using wta as weights. 
        # The wta contains the indexes of kernels with the highest activations... but it doesn't make sense...hold on.
        myprint(wta[0])
        
        yx = F.conv2d(
            x.transpose(0, 1),
            wta.transpose(0, 1), # the wta are the weights associated to all the pre activations. 
            padding=self.padding,
            stride=self.dilation,
            dilation=self.stride,
            groups=1
        ).transpose(
            0, 1
        )

        yx = yx[:, :, :self.kernel_size[0], :self.kernel_size[1]]

        yu = torch.sum(torch.mul(wta, pre_x), dim=(0, 2, 3))

        delta_weight = yx - yu.view(-1, 1, 1, 1) * weight

        # ---Normalize---#
        nc = torch.abs(delta_weight).amax()
        delta_weight.div_(nc + 1e-30)

        # print((self.rad.nelement() * self.rad.element_size()+self.lr.nelement() * self.lr.element_size()+yx.nelement() * yx.element_size()+x.nelement() * x.element_size()+wta.nelement() * wta.element_size()+yx.nelement() * yx.element_size()+yu.nelement() * yu.element_size()+delta_weight.nelement() * delta_weight.element_size()+self.weight.nelement() * self.weight.element_size())/ 1024**3)
        # print(yx.shape, x.shape, pre_x.shape, wta.shape, yx.shape, yu.shape, delta_weight.shape)
        return delta_weight

    def delta_weight_linear(
            self,
            x: torch.Tensor,
            pre_x: torch.Tensor,
            wta: torch.Tensor, ) -> torch.Tensor:
        """
        Compute the change of weights with the linear calculation
        Parameters
        ----------
        x : torch.Tensor
            x. Input (batch_size, in_features).
        pre_x : torch.Tensor
            pre_x. Linear transformation of the input (batch_size, in_features).
        wta : torch.Tensor
            wta. Winner take all (batch_size, in_features).
        Returns
        -------
            delta_weight : torch.Tensor
        """
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                           dilation=self.dilation)
        n_parameters_per_kernel = patches.size(1)
        patches = patches.swapaxes(0, 1).reshape(n_parameters_per_kernel, -1).swapaxes(0, 1)

        wta_flat = wta.permute(1, 0, 2, 3).reshape(self.out_channels, -1).t()

        yx_flat = torch.matmul(wta_flat.t(), patches)  # Overlap between winner take all and inputs

        yu_flat = torch.multiply(wta_flat, wta_flat)
        yu_flat = torch.sum(yu_flat.t(), dim=1).unsqueeze(1)
        # Overlap between preactivation and winner take all
        # Results are summed over batches, resulting in a shape of (output_size,)
        yuw_flat = yu_flat.view(-1, 1) * self.weight.reshape(self.out_channels, -1)
        delta_weight = yx_flat - yuw_flat
        # ---Normalize---#
        nc = torch.abs(delta_weight).amax()  # .amax(1, keepdim=True)
        delta_weight.div_(nc + 1e-30)

        return delta_weight.reshape(self.out_channels, self.in_channels, *self.kernel_size)

    def plasticity(
            self,
            x_groups: torch.Tensor,
            pre_x_groups: torch.Tensor = None,
            wta_groups: torch.Tensor = None) -> None:
        """
        Update weight and bias accordingly to the plasticity computation
        Parameters
        ----------
        x_groups : torch.Tensor
            x. Input (batch_size, in_features).
        pre_x_groups : torch.Tensor
            pre_x. Conv2d transformation of the input (batch_size, in_features).
        wta_groups : torch.Tensor
            wta. Winner take all (batch_size, in_features).

        """
        weight_groups = torch.chunk(self.weight, self.groups, dim=0)
        self.delta_w = torch.cat([self.delta_weight(x_group, pre_x_group, wta_group, weight_group) for
                                  x_group, pre_x_group, wta_group, weight_group in
                                  zip(x_groups, pre_x_groups, wta_groups, weight_groups)])

        # self.update(delta_w)

        if self.bias is not None:
            self.delta_b = self.delta_bias(wta_groups)
    
    def average_deltas(self):
        # returns the average normalized weight change
        # self.delta_w.shape = torch.Size([1536, 384, 3, 3])
        device = DEVICE
        summed_kernels = torch.sum(self.delta_w, dim=(1,2,3)).to(device)
        average = summed_kernels / summed_kernels.shape[0]
        average_norm = average / max(average)
        return average_norm

    def check_threshold(self, current_kernels_avg):
        # returns a mask which identifies 
        device = DEVICE
        mask = (current_kernels_avg > self.avg_deltas_layer).type(torch.uint8).to(device)
        if self.temp == 1:
            print("CHECK_THRESHOLD: ", mask[0], current_kernels_avg.cpu()[0], self.avg_deltas_layer.cpu()[0])
            self.temp = 2
        return mask
    
    def get_lower_lr_mask(self, threshold_mask):
        # this function returns a one hot tensor where the cell to 0 are the ones where we don't need to decrease the lr and the cells with 1 are the ones where we have to
        # decrease the lr
        device = DEVICE
        topk_mask = torch.zeros(len(threshold_mask)).to(device)
        topk_mask[self.top_acts_layer] = 1
        final_mask = topk_mask + threshold_mask
        if self.temp == 2:
            print("topk_mask ", topk_mask.cpu())
            print("threshold_mask ", threshold_mask.cpu())
            print("final_mask", final_mask.cpu())
            self.temp = 3
        final_mask = (final_mask == 2 ).type(torch.uint8)
        return final_mask
    def get_higher_lr_mask(self):
        # this function returns a one hot tensor where the cell to 0 are the on s where we don't need to increase the lr and the cells with 1 are the ones where we have to
        # increase the lr
        # to do we take a one hot encoding of the cells representing with one the top k kernels
        device = DEVICE
        topk_mask = torch.zeros(self.out_channels).to(device)
        topk_mask[self.top_acts_layer] = 1

        # since we need to consider only the cells not among the top k kernels we calculate the inverse
        not_topk_mask = (topk_mask == 0).type(torch.uint8)
        if self.temp == 2:
            print("topk_mask ", topk_mask.cpu())
            print("not_topk_mask ", not_topk_mask.cpu())
            self.temp = 3
        return not_topk_mask

    def update(self) -> None:

        # nb_train è un valore che corrisponde sempre a out_channels dato che non è mai stato inizializzato perchè nei preset non 
        # ci sta.

        if self.nb_train > 0:
            # self.delta_w[self.mask] = 0
            #print(self.lr.shape)
            # !!! self.weight.shape is always equal to self.nb_train
################################################################################
            if self.temp == 0:
                print("self.avg_deltas_layer: ", type(self.avg_deltas_layer))
                print("self.top_acts_layer: ", type(self.top_acts_layer))
                self.temp = 1
            if self.cl_hyper["cf_sol"] and self.avg_deltas_layer is not None and self.top_acts_layer is not None:
                lower_lr = self.cl_hyper["low_lr"] # means that we reduce it of x%
                higher_lr = self.cl_hyper["high_lr"] # means that we increase it of x%
                # first we find the average weight change per kernel
                current_kernels_avg = self.average_deltas()
                # find a tensor which flags the kernels that break the threshold
                threshold_mask = self.check_threshold(current_kernels_avg)

                # create a mask of kernels which break the threshold and are in the top k kernels, will be used to reduce the learning rate 
                lower_lr_mask = self.get_lower_lr_mask(threshold_mask)
                # 10% decrease
                lower_lr_mask = -lower_lr*lower_lr_mask

                # create a mask which contains 1s in the cells where the learning rate has to be increased and zeros otherwise.
                # basically it has 1s where the cells are not among the top k kernels. 
                higher_lr_mask = self.get_higher_lr_mask()
                # 10% increase
                higher_lr_mask = higher_lr*higher_lr_mask

                lr_mask = 1 + higher_lr_mask+lower_lr_mask
                lr_mask = lr_mask.view(self.lr.shape[0], 1, 1, 1)
                # return a mask of the learning rate which limits the learning where the threshold was broken and increases it where it wasn't.
                if self.temp == 3:
                    
                    print("lower_lr_mask ", lower_lr_mask.cpu())
                    print("higher_lr_mask ", higher_lr_mask.cpu())
                    print("lr_mask ", lr_mask.cpu())
                    print(self.lr.shape)
                    print("self.lr", self.lr[0])
                    self.temp = 4
                #lr_mask = lr_mask.to("cuda:0")
                self.weight[:self.nb_train].add_(self.lr *lr_mask*self.delta_w[:self.nb_train])
################################################################################################
            else:
                self.weight[:self.nb_train].add_(self.lr*self.delta_w[:self.nb_train])
                #self.weight[:self.nb_train].add_(self.lr*self.delta_w[:self.nb_train])
            if self.temp == 4:
                
                self.temp = 5
            # self.weight = self.weight * self.mask
            self.was_update = True
            if self.bias is not None:
                if self.lr_adaptive:
                    self.bias.add_(self.lr[:, 0, 0, 0] * self.lrb * self.delta_b)
                else:
                    self.bias.add_(self.lr * self.lrb * self.delta_b)
                # self.bias.clip_(-1, 0)
            self.update_lr()

    def update_lr(self) -> None:
        if self.lr_adaptive:
            norm = self.get_radius()

            nc = 1e-10

            # lr_amplitude = next(self.lr_scheduler)

            lr_amplitude = self.lr_scheduler_config['lr']

            lr = lr_amplitude * torch.pow(torch.abs(norm - torch.ones_like(norm)) + nc,
                                          self.lr_scheduler_config['power_lr'])

            # lr = lr.clip(max=lr_amplitude)

            self.stat[0] = lr.clone()

            self.lr = einops.repeat(lr, 'o -> o i k k2', k=self.kernel_size[0], k2=self.kernel_size[0],
                                    i=self.in_channels // self.groups)
        else:
            self.lr = next(self.lr_scheduler)
            self.stat[0] = self.lr

    def get_radius(self):
        if self.was_update:
            weight = self.weight.view(self.weight.shape[0], -1)
            self.rad = torch.linalg.norm(weight, dim=1, ord=self.lebesgue_p)
            self.was_update = False
        return self.rad

    def extra_repr2(self):
        s = '{in_channels}, {out_channels}, lebesgue_p={lebesgue_p}, pruning={mask_thsd}, kernel_size={kernel_size}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.stride != (1,) * len(self.stride):
            s += ', stride={stride}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def extra_repr(self):
        return self.extra_repr2()

    def __label__(self):
        s = '{in_channels}{out_channels}{lebesgue_p}{kernel_size}'
        if self.padding != (0,) * len(self.padding):
            s += '{padding}'
        if self.stride != (1,) * len(self.stride):
            s += '{stride}'
        if self.dilation != (1,) * len(self.dilation):
            s += '{dilation}'
        if self.groups != 1:
            s += '{groups}'
        if self.padding_mode != 'zeros':
            s += '{padding_mode}'
        return 'H' + s.format(**self.__dict__)

    def radius(self) -> float:
        """
        Returns
        -------
        radius : float
        """
        meanb = torch.mean(self.bias) if self.bias is not None else 0.
        stdb = torch.std(self.bias) if self.bias is not None else 0.
        weight = self.weight.view(self.weight.shape[0], -1)
        mean = torch.mean(weight, axis=1)
        mean_weight = torch.mean(mean)
        std_weigh = torch.std(weight)

        norm = self.get_radius()
        self.stat[1] = norm
        mean_radius = torch.mean(norm)
        std_radius = torch.std(norm)
        max_radius = torch.amax(torch.abs(norm - mean_radius * torch.ones_like(norm)))
        mean2_radius = torch.mean(torch.abs(norm - mean_radius * torch.ones_like(norm)))

        return 'MB:{mb:.3e}/SB:{sb:.3e}/MW:{m:.3e}/SW:{s:.3e}/MR:{mean:.3e}/SR:{std:.3e}/MeD:{mean2:.3e}/MaD:{max:.3e}'.format(
            mb=meanb,
            sb=stdb,
            m=mean_weight,
            s=std_weigh,
            mean=mean_radius,
            std=std_radius,
            mean2=mean2_radius,
            max=max_radius) + '\n' + self.stat_wta() + '\n'

    def convergence(self) -> float:
        """
        Returns
        -------
        convergence : float
            Metric of convergence as the nb of filter closed to 1
        """
        weight = self.weight.view(self.weight.shape[0], -1)

        norm = self.get_radius()
        # mean_radius = torch.mean(norm)
        conv = torch.mean(torch.abs(norm - torch.ones_like(norm)))

        R1 = torch.sum(torch.abs(norm - torch.ones_like(norm)) < 1e-2)
        return float(conv.cpu()), int(R1.cpu())


class HebbHardKrotovConv2d(HebbHardConv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            lebesgue_p: int,
            kernel_size: int,
            weight_distribution: str,
            weight_range: float,
            pre_triangle: bool,
            mask_thsd: float,
            lr_scheduler: Generator,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            padding_mode: str = "zeros",
            bias: bool = False,
            delta: float = 0.05,
            ranking_param: int = 2,
            nb_train: int = None,
    ) -> None:
        """
       Krotov implementation from the HardLinear class
       """

        super(HebbHardKrotovConv2d, self).__init__(in_channels, out_channels, lebesgue_p, kernel_size,
                                                   weight_distribution,
                                                   weight_range, pre_triangle, mask_thsd, lr_scheduler, stride, padding,
                                                   dilation, groups, padding_mode, bias, nb_train)

        self.delta = delta
        self.ranking_param = ranking_param
        self.lebesgue_p = lebesgue_p

        self.stat = torch.zeros(4, out_channels)

    def extra_repr(self):
        s = ', ranking_param=%s, delta=%s' % (self.ranking_param, self.delta)
        return self.extra_repr2() + s

    def __label__(self):
        s = '{in_channels}{out_channels}{lebesgue_p}{kernel_size}{ranking_param}{delta}'
        if self.padding != (0,) * len(self.padding):
            s += '{padding}'
        if self.stride != (1,) * len(self.stride):
            s += '{stride}'
        if self.dilation != (1,) * len(self.dilation):
            s += '{dilation}'
        if self.groups != 1:
            s += '{groups}'
        if self.padding_mode != 'zeros':
            s += '{padding_mode}'
        return 'HK' + s.format(**self.__dict__)

    def get_wta(self, pre_x: torch.Tensor, group_id: int = 0) -> torch.Tensor:
        """
        Compute the krotov winner take all
        ----------
        pre_x : torch.Tensor
            pre_x
        Returns
        -------
        wta : torch.Tensor
            preactivation or the current of the hebbian layer
        """
        batch_size, out_channels, height_out, width_out = pre_x.shape
        #print(pre_x.shape)
        #print(pre_x)
        pre_x_flat = pre_x.transpose(0, 1).reshape(out_channels, -1)
        _, ranks = pre_x_flat.sort(descending=True, dim=0)
        wta = nn.functional.one_hot(pre_x_flat.argmax(dim=0), num_classes=pre_x_flat.shape[0]).to(
            torch.float)
        self.stat[2, group_id * self.out_channels_groups: (group_id + 1) * self.out_channels_groups] += wta.sum(0).cpu()

        wta = wta - self.delta * nn.functional.one_hot(ranks[self.ranking_param - 1], num_classes=pre_x_flat.shape[0])

        self.stat[3] += torch.histc(torch.tensor(ranks[self.ranking_param - 1]), bins=self.out_channels, min=0,
                                    max=self.out_channels - 1).cpu()

        wta = wta.transpose(0, 1).view(
            out_channels, batch_size, height_out, width_out
        ).transpose(0, 1)


        return wta


class HebbSoftConv2d(HebbHardConv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            lebesgue_p: int,
            kernel_size: int,
            weight_distribution: str,
            weight_range: float,
            pre_triangle: bool,
            mask_thsd: float,
            lr_scheduler: Generator,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            padding_mode: str = "zeros",
            bias: bool = False,
            activation_fn: str = 'exp',
            t_invert: float = 12,
            nb_train: int = None,
    ) -> None:
        """
        Soft implementation from the HardConv2d class
        """

        super(HebbSoftConv2d, self).__init__(in_channels, out_channels, lebesgue_p, kernel_size, weight_distribution,
                                             weight_range, pre_triangle, mask_thsd, lr_scheduler, stride, padding,
                                             dilation, groups, padding_mode, bias, nb_train)

        self.stat = torch.zeros(6, out_channels)

        self.activation_fn = activation_fn
        self.t_invert = torch.tensor(t_invert)

        if bias:
            self.register_buffer('bias', torch.ones(out_channels) \
                                 * torch.log(torch.tensor(1 / out_channels)) / self.t_invert)
            self.delta_b = torch.zeros_like(self.bias)

        self.lrb = torch.tensor(0.1 / t_invert)

    def extra_repr(self):
        s = ', t_invert=%s, bias=%s, lr_bias=%s' % (
        float(self.t_invert), not self.bias is None, round(float(self.lrb), 4))
        s += ', activation=%s' % self.activation_fn
        return self.extra_repr2() + s

    def __label__(self):
        s = '{in_channels}{out_channels}{lebesgue_p}{kernel_size}{t_invert}'
        if self.padding != (0,) * len(self.padding):
            s += '{padding}'
        if self.stride != (1,) * len(self.stride):
            s += '{stride}'
        if self.dilation != (1,) * len(self.dilation):
            s += '{dilation}'
        if self.groups != 1:
            s += '{groups}'
        if self.padding_mode != 'zeros':
            s += '{padding_mode}'
        return 'S' + s.format(**self.__dict__)

    def get_wta(self, pre_x: torch.Tensor, group_id: int = 0) -> torch.Tensor:
        """
        Compute the soft winner take all
        ----------
        pre_x : torch.Tensor
            pre_x
        Returns
        -------
        wta : torch.Tensor
            preactivation or the current of the hebbian layer
        We take the pre activations and pass them though the activation function which returns a set of values 
        between 0 and 1. So instead of adjusting the weights of the neuron with the highest pre activation we 
        update the weights of all the neurons according to their activation.
        """
        wta = activation(pre_x, t_invert=self.t_invert, activation_fn=self.activation_fn, normalize=True, dim=1)
        self.stat[2, group_id * self.out_channels_groups: (group_id + 1) * self.out_channels_groups] += wta.sum(
            (0, 2, 3)).cpu()

        return wta

    def delta_bias(self, wta: torch.Tensor) -> None:
        """
        Compute the change of Bias
        Parameters
        ----------
        wta : torch.Tensor
            wta. Winner take all (batch_size, in_features).
        """
        batch_size, _, w, h = wta.shape
        if self.activation_fn == 'exp':
            ebb = torch.exp(self.t_invert * self.bias)  # e^(bias*t_invert)
            # ---Compute change of bias---#
            delta_bias = (torch.sum(wta, dim=(0, 2, 3)) - ebb * batch_size * w * h) / ebb
        elif self.activation_fn == 'relu':
            delta_bias = (
                        torch.sum(wta, dim=(0, 2, 3)) - batch_size * w * h * self.bias - batch_size)  # eta *    (y-w-1)

        nc = torch.abs(delta_bias).amax()  # .amax(1, keepdim=True)
        delta_bias.div_(nc + 1e-30)

        return delta_bias


class HebbSoftKrotovConv2d(HebbSoftConv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            lebesgue_p: int,
            kernel_size: int,
            weight_distribution: str,
            weight_range: float,
            pre_triangle: bool,
            mask_thsd: float,
            lr_scheduler: Generator,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            padding_mode: str = "zeros",
            bias: bool = False,
            ranking_param: int = 2,
            delta: float = 4,
            activation_fn: str = 'exp',
            t_invert: float = 12,
            nb_train: int = None,
            avg_deltas_layer: torch.tensor = None,
            top_acts_layer: list = None,
            cl_hyper: dict = {}
    ) -> None:
        """
        Krotov implementation from the SoftConv2d class
        """

        super(HebbSoftKrotovConv2d, self).__init__(in_channels, out_channels, lebesgue_p, kernel_size,
                                                   weight_distribution,
                                                   weight_range, pre_triangle, mask_thsd, lr_scheduler, stride, padding,
                                                   dilation, groups, padding_mode, bias, activation_fn, t_invert,
                                                   nb_train)

        self.delta = delta
        self.ranking_param = ranking_param
        self.m_winner = []
        self.m_anti_winner = []
        self.mode = 0

        self.avg_deltas_layer = avg_deltas_layer
        self.top_acts_layer = top_acts_layer
        self.cl_hyper = cl_hyper
        # self.stat = torch.zeros(5, out_channels)

    def extra_repr(self):
        s = ', t_invert=%s, bias=%s, lr_bias=%s' % (
        float(self.t_invert), not self.bias is None, round(float(self.lrb), 4))
        s += ', ranking_param=%s, delta=%s' % (self.ranking_param, self.delta)
        s += ', activation=%s' % self.activation_fn
        return self.extra_repr2() + s

    def __label__(self):
        s = '{in_channels}{out_channels}{lebesgue_p}{kernel_size}{t_invert}'
        if self.padding != (0,) * len(self.padding):
            s += '{padding}'
        if self.stride != (1,) * len(self.stride):
            s += '{stride}'
        if self.dilation != (1,) * len(self.dilation):
            s += '{dilation}'
        if self.groups != 1:
            s += '{groups}'
        if self.padding_mode != 'zeros':
            s += '{padding_mode}'
        return 'SK' + s.format(**self.__dict__)

    def get_wta(self, pre_x: torch.Tensor, group_id: int = 0) -> torch.Tensor:
        """
        Compute the soft krotov winner take all
        ----------
        pre_x : torch.Tensor
            pre_x
        Returns
        -------
        wta : torch.Tensor
            preactivation or the current of the hebbian layer

        pre_x shape [batch_size, out_channels, height_out, width_out]
        The number of output channels corresponds to the number of feature maps, which corresponds to the 
        number of kernels in the filter.

        In the given tensor shape [10, 96, 32, 32]:

        - Batch Size (10): This means that the tensor contains 10 samples (or images) that are being processed together.
        -Channels (96): Each sample has 96 channels, often corresponding to feature maps in the context of convolutional neural networks.
        -Height and Width (32, 32): Each channel has spatial dimensions of 32x32.
        wta shape: 


        """
        batch_size, out_channels, height_out, width_out = pre_x.shape

        #f1 = open('pre_x.txt', "w")
        #f1.write(np.array2string(pre_x.cpu()))

        #print("PRE_X:", pre_x.shape)


        pre_x_flat = pre_x.transpose(0, 1).reshape(out_channels, -1)

        """
        
        pre_x: Original shape: torch.Size([10, 96, 32, 32])
        Transposed shape: torch.Size([96, 10, 32, 32])
        pre_x_flat: Reshaped shape: torch.Size([96, 10240])
        pre_x_flat : [out_channels, height*width*batch_size]
        The question now becomes, why are we putting it to wta shape? Like why is wta of shape [96, 10240]?? 
        So wta is of shape [out_channels, height*width*batch_size].
        """
        #np.savetxt('pre_x_flat.txt', pre_x_flat.cpu().numpy())
        #print("PRE_X_FLAT:", pre_x_flat.shape)

        # here we basically take all the pre activations and feed them thouigh a softmax activation function to obtain 
        # values between 0 and 1 and which all sum up to 1
        wta = activation(pre_x_flat, t_invert=self.t_invert, activation_fn=self.activation_fn, normalize=True, dim=0)

        self.stat[2, group_id * self.out_channels_groups: (group_id + 1) * self.out_channels_groups] += wta.sum(1).cpu()

        # torch.arange creates a tensor containing indices from 0 (if no other starting argument is provided)
        # and ends the list of indices at pre_x_flat.size(1), which in this case would correspond to 10240.
        batch_indices = torch.arange(pre_x_flat.size(1))

        if self.mode == 0:
            wta = -wta
            #if xyz != 11: 
            #print(wta.shape)
            #print("WTA[0:20][0:10] :", wta[0:20][0:10])
            #np.savetxt('wta.txt', wta.numpy())

            #xyz = 11
            # _, ranking_indices = pre_x_flat.topk(1, dim=0)
            # ranking_indices = ranking_indices[0, batch_indices]

            # Basically with this operation we extract the indexes of the max values per every 10240 values, 
            # so for all the values at pos 0 we find in which among the 96 columns the maximum is, then we do it 
            # for the value at position 1, and so on. So we will have as a result a vector which has 10240 values with range [0, 95]
            # Each element in ranking_indices corresponds to the index of the channel (out of 96) that 
            # has the highest value for a specific flattened spatial position.
            ranking_indices = torch.argmax(pre_x_flat, dim=0)
            #print("RANKING: ", ranking_indices)


            """
            [
            [ 10,  20,  30,  40],
            [ 50,  60,  70,  80],
            [ 90, 100, 110, 120]
            ]

            row_indices = [0, 1, 2]
            col_indices = [1, 2, 3]

            Selecting Elements:
            tensor[row_indices, col_indices] selects elements at positions:
            (0, 1) -> 20
            (1, 2) -> 70
            (2, 3) -> 120

            So in our case we do wta[ranking_indices, batch_indices] because we want to consider all the rows that allow us to extract 
            the maximum value, so the ranking_indices contains the columns while the batch indices says we have to do it for all the samples.
            wta is of shape [96, 10240], so we have 96 rows and 10240 columns, so if we need to select 10240 values, we need to have a set of 10240 row
            values and 10240 column values. That's why we do wta[ranking_indices, batch_indices] [10240 values, 10240 values]

            """
            # print("BATCH_INDICES:", batch_indices.shape)
            # print("ranking_indices:", ranking_indices.shape)
            # print("wta[ranking_indices, batch_indices]: ", wta[ranking_indices, batch_indices].shape)
            # print("wta[ranking_indices, batch_indices].mean(): ", wta[ranking_indices, batch_indices].mean())


            #here we inverse the values of the wta vector
            # Here, the values corresponding to the winning channels (ranking_indices) and their batch positions (batch_indices) are negated. 
            # This operation updates the winning values, possibly to enforce a competitive learning mechanism.
            wta[ranking_indices, batch_indices] = -wta[ranking_indices, batch_indices]

            #here we calculate the the mean of the vector containing the winning values
            #m_winner contains the mean of the winners for every batch, so it's a list
            self.m_winner.append(wta[ranking_indices, batch_indices].mean().cpu())

            #m_anti winner is the anti-hebbian value of the given batch (so it is calculated per batch)
            self.m_anti_winner.append(1 - self.m_winner[-1])

            # self.t_invert += 0.1 * (0.7 - np.mean(self.m_winner[-10:]))**2 * (0.7-self.m_winner[-1])
            # print(self.t_invert, np.mean(self.m_winner[-5:]), self.m_winner[-1])

        if self.mode == 1:

            # _, ranking_indices = pre_x_flat.topk(self.ranking_param, dim=0)

            """
            Here we use the flattened pre_x tensor, which is flattened with respect to dimension 96, while the batches, the height an dthe width are
            all multiplies together and placed in tensors of size 10240. In particular we sort the pre_x_flat tensor wrt the dim 0, so like we create tensors
            which have the 96 tensors created by considering the sorted values, so pre_x_flat[0][0] has taken the highest value among all the other 95 tensors considering
            only elemnts at position 0. Descending so from high to low. tensor.sort returns both the tensor with the ordered values and a new tensor which contains
            the indexes of the elements in the original tensor ( in this case the indexes are associated with ranks and the new ordered tensor with _ ).
            """
            _, ranks = pre_x_flat.sort(descending=True, dim=0)
            # print(torch.tensor(ranking_indices[0, batch_indices]).shape, torch.histc(torch.tensor(ranking_indices[0, batch_indices]), bins=96))

            """
            A this point we have ranks which is a tensor of shape [96, 10240]. From ranks we consider the last ranking_param and extract it from wta considering all the batch_indices, 
            from this tensor we calculate the mean.  Then we get the winner for the current batch from wta and calculate its mean.
            But where are m_winner and m_anti_winner used?? Ok so they are used for the calculation of the radius which is among the "info" when calling the trian unsup/sup function.

            """
            self.m_anti_winner.append(
                wta[ranks[self.ranking_param - 1], batch_indices].mean().cpu())
            self.m_winner.append(wta[ranks[0], batch_indices].mean().cpu())

            self.stat[3, group_id * self.out_channels_groups: (group_id + 1) * self.out_channels_groups] += torch.histc(
                torch.tensor(ranks[self.ranking_param - 1]), bins=self.out_channels, min=0,
                max=self.out_channels - 1).cpu()

            # print(wta[ranking_indices[self.ranking_param - 1, batch_indices], batch_indices].mean())
            """
            Finally we take the last tensor among the ones to be considered and update its value by multiplying it with a delta.
            Where is this delta defined?? Ok so delta is preset to 4, so we multiply the values of the last ranking element by -4, 
            pretty big decrease but ok. 
            """
            wta[ranks[
                    self.ranking_param - 1], batch_indices] *= -self.delta  # / torch.mean(torch.tensor(self.m_anti_winner[-100:]))
            # -self.delta#-(
            # self.delta / torch.mean(torch.tensor(self.m_anti_winner[-10:])))
            # print(wta[ranking_indices[self.ranking_param - 1, batch_indices], batch_indices].mean())

        if self.mode == 2:

            """
            Here we consider the pre_x_flat tensor with shape [96, 10240] and take the first topk elements, where k is given by ranking_param and the topk 
            is done with respect to the dim 0, so like the previous case we consider the highest value  among the 96 available for each index of the 10240 elements. 
            Again, we store the topk tensor in _ and the indexes of the original tensor in ranking_indices.
            """
            _, ranking_indices = pre_x_flat.topk(self.ranking_param, dim=0)

            """
            At this point we take the wta vector and consider only the ranking indices, so we extract only the topk values and extract the last one considering all the 
            batch_indices values, with these we calculate the mean and store it in m_anti_winner.
            """
            self.m_anti_winner.append(
                wta[ranking_indices[self.ranking_param - 1, batch_indices], batch_indices].mean().cpu())
            """
            Again, take the first tensor among the topk and calculate the mean along the batch_indices values to store it in m_winner.
            """
            self.m_winner.append(wta[ranking_indices[0, batch_indices], batch_indices].mean().cpu())

            # print(wta[ranking_indices[self.ranking_param - 1, batch_indices], batch_indices].mean())
            """
            Finally, take the last tensor among the topk in wta and subtract to it the delta, which is initialized to 4, effectively reducing the 
            strength of the activation by a constant of 4.
            """
            wta[ranking_indices[self.ranking_param - 1, batch_indices], batch_indices] = -self.delta

        wta = wta.view(
            out_channels, batch_size, height_out, width_out
        ).transpose(0, 1)
        
        return wta

    def radius(self) -> float:
        """
        Returns
        -------
        radius : float
        """
        meanb = torch.mean(self.bias) if self.bias is not None else 0.
        stdb = torch.std(self.bias) if self.bias is not None else 0.
        weight = self.weight.view(self.weight.shape[0], -1)
        mean = torch.mean(weight, axis=1)
        mean_weight = torch.mean(mean)
        std_weigh = torch.std(weight)

        norm = torch.linalg.norm(weight, dim=1, ord=self.lebesgue_p)
        self.stat[1] = norm
        mean_radius = torch.mean(norm)
        std_radius = torch.std(norm)
        max_radius = torch.amax(torch.abs(norm - mean_radius * torch.ones_like(norm)))
        mean2_radius = torch.mean(torch.abs(norm - mean_radius * torch.ones_like(norm)))

        m_winner = torch.mean(torch.tensor(self.m_winner[-10:]))
        m_anti_winner = torch.mean(torch.tensor(self.m_anti_winner[-10:]))

        return 'MB:{mb:.3e}/SB:{sb:.3e}/MW:{m:.3e}/SW:{s:.3e}/MR:{mean:.3e}/SR:{std:.3e}/MeD:{mean2:.3e}/MaD:{max:.3e}/MW:{m_winner:.3f}/MAW:{m_anti_winner:.3f}'.format(
            mb=meanb,
            sb=stdb,
            m=mean_weight,
            s=std_weigh,
            mean=mean_radius,
            std=std_radius,
            mean2=mean2_radius,
            max=max_radius,
            m_winner=m_winner,
            m_anti_winner=m_anti_winner
        ) + '\n' + self.stat_wta() + '\n'


def select_Conv2d_layer(params, avg_deltas_layer=None, acts_layer=None, cl_hyper={}) -> Union[HebbHardConv2d, HebbHardKrotovConv2d, HebbSoftConv2d, HebbSoftKrotovConv2d]:
    """
    Select the appropriate from a set of params
    ----------
    params : torch.Tensor
        wta. Winner take all (batch_size, in_features).
    Returns
        -------
        layer : bio
            preactivation or the current of the hebbian layer

    """
    if params['softness'] == 'hard':
        layer = HebbHardConv2d(
            in_channels=params['in_channels'],
            out_channels=params['out_channels'],
            lebesgue_p=params['lebesgue_p'],
            kernel_size=params['kernel_size'],
            weight_distribution=params['weight_init'],
            weight_range=params['weight_init_range'],
            pre_triangle=params['pre_triangle'],
            mask_thsd=params['mask_thsd'],
            lr_scheduler=params['lr_scheduler'],
            stride=params['stride'],
            padding=params['padding'],
            dilation=params['dilation'],
            groups=params['groups'],
            padding_mode=params['padding_mode'],
            bias=params['add_bias'],
            nb_train=params['nb_train'], 
            # avg_deltas_layer = avg_deltas_layer,
            # acts_layer = acts_layer
        )
    elif params['softness'] == 'hardkrotov':
        layer = HebbHardKrotovConv2d(
            in_channels=params['in_channels'],
            out_channels=params['out_channels'],
            lebesgue_p=params['lebesgue_p'],
            kernel_size=params['kernel_size'],
            weight_distribution=params['weight_init'],
            weight_range=params['weight_init_range'],
            pre_triangle=params['pre_triangle'],
            mask_thsd=params['mask_thsd'],
            lr_scheduler=params['lr_scheduler'],
            stride=params['stride'],
            padding=params['padding'],
            dilation=params['dilation'],
            groups=params['groups'],
            padding_mode=params['padding_mode'],
            bias=params['add_bias'],
            delta=params['delta'],
            ranking_param=params['ranking_param'],
            nb_train=params['nb_train'], 
            # avg_deltas_layer = avg_deltas_layer,
            # acts_layer = acts_layer
        )
    elif params['softness'] == 'soft':
        layer = HebbSoftConv2d(
            in_channels=params['in_channels'],
            out_channels=params['out_channels'],
            lebesgue_p=params['lebesgue_p'],
            kernel_size=params['kernel_size'],
            weight_distribution=params['weight_init'],
            weight_range=params['weight_init_range'],
            pre_triangle=params['pre_triangle'],
            mask_thsd=params['mask_thsd'],
            lr_scheduler=params['lr_scheduler'],
            stride=params['stride'],
            padding=params['padding'],
            dilation=params['dilation'],
            groups=params['groups'],
            padding_mode=params['padding_mode'],
            bias=params['add_bias'],
            activation_fn=params['soft_activation_fn'],
            t_invert=params['t_invert'],
            nb_train=params['nb_train'], 
            # avg_deltas_layer = avg_deltas_layer,
            # acts_layer = acts_layer
            )
    elif params['softness'] == 'softkrotov':
        layer = HebbSoftKrotovConv2d(
            in_channels=params['in_channels'],
            out_channels=params['out_channels'],
            lebesgue_p=params['lebesgue_p'],
            kernel_size=params['kernel_size'],
            weight_distribution=params['weight_init'],
            weight_range=params['weight_init_range'],
            pre_triangle=params['pre_triangle'],
            mask_thsd=params['mask_thsd'],
            lr_scheduler=params['lr_scheduler'],
            stride=params['stride'],
            padding=params['padding'],
            dilation=params['dilation'],
            groups=params['groups'],
            padding_mode=params['padding_mode'],
            bias=params['add_bias'],
            delta=params['delta'],
            ranking_param=params['ranking_param'],
            activation_fn=params['soft_activation_fn'],
            t_invert=params['t_invert'],
            nb_train=params['nb_train'],
            avg_deltas_layer = avg_deltas_layer,
            top_acts_layer = acts_layer, 
            cl_hyper = cl_hyper


        )
    return layer
