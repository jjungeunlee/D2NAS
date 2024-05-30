import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import operations as ops
from train_utils import drop_path


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, ks=None, ds=None, ps=None, dims=None):
        super(Cell, self).__init__()
        if reduction_prev:
            self.preprocess0 = ops.FactorizedReduce(C_prev_prev, C, dims=dims)
        else:
            self.preprocess0 = ops.ReLUConvBN(C_prev_prev, C, 1, 1, 0, dims=dims)
        self.preprocess1 = ops.ReLUConvBN(C_prev, C, 1, 1, 0, dims=dims)
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction, ks=ks, ds=ds, ps=ps, dims=dims)

    def _compile(self, C, op_names, indices, concat, reduction, ks=None, ds=None, ps=None, dims=None):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = ops.OPS[name](C, stride, True, ks=ks, ds=ds, ps=ps, dims=dims)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, ops.Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, ops.Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class Network(nn.Module):

    def __init__(self, C_in, C_pre, num_classes, layers, genotype, config_kwargs, ks=None, ds=None, dims=None):
        super(Network, self).__init__()
        self.no_reduction = config_kwargs['no_reduction']
        self.activation = config_kwargs['activation']
        self.remain_shape = config_kwargs['remain_shape']
        self.squeeze = config_kwargs['squeeze']
        self.pool_k = config_kwargs['pool_k']
        self.dims = dims

        self._layers = layers
        stem_multiplier = 1

        C_curr = stem_multiplier * C_pre
        k, d, p = (ks[0], ds[0], ks[0]//2 * ds[0]) if ks is not None else (3, 1, 1)
        self.stem = nn.Sequential(
            ops.get_conv_layer(C_in, C_curr, kernel_size=k, dilation=d, padding=p, dims=dims),
            ops.get_bn_layer(C_curr, dims=dims)
        )
        ks = deepcopy(ks[1:]) if ks is not None else None # ks[0] was for self.stem
        ds = deepcopy(ds[1:]) if ds is not None else None # ds[0] was for self.stem
        ps = [k // 2 * d for k, d in zip(ks, ds)] if ks is not None else None
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C_pre
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if self.no_reduction == False and i in [layers//3, max(3, 2*layers//3)]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, ks=ks, ds=ds, ps=ps, dims=dims)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr
        self.global_pooling = ops.get_pool_layer(type='adaptiveavg', output_size=1, dims=dims)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)

        if self.remain_shape:
            if self.dims == 1:
                s1 = s1.permute(0, 2, 1)
                s1 = self.classifier(s1)
                s1 = s1.permute(0, 2, 1)
            else:
                s1 = s1.permute(0, 2, 3, 1)
                s1 = self.classifier(s1)
                s1 = s1.permute(0, 3, 1, 2)
        else:
            if self.dims == 1:
                s1 = self.global_pooling(s1) if self.pool_k > 0 else s1.mean(-1)
                s1 = s1.view(s1.size(0),-1)
                s1 = self.classifier(s1)
            else:
                s1 = self.global_pooling(s1) if self.pool_k > 0 else s1.view(s1.size(0), s1.size(1), -1).mean(-1)
                s1 = s1.view(s1.size(0),-1)
                s1 = self.classifier(s1)

        if self.squeeze:
            s1 = s1.squeeze()

        if self.activation == 'sigmoid':
            logits = torch.sigmoid(s1)
        elif self.activation == 'softmax':
            logits = F.log_softmax(s1, dim=1)
        else:
            logits = s1

        return logits