import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import operations as ops
from genotypes import PRIMITIVES


class MixedOp(nn.Module):

    def __init__(self, C, stride, switch, p, dims):
        super(MixedOp, self).__init__()
        self.m_ops = nn.ModuleList()
        self.p = p
        for i in range(len(switch)):
            if switch[i]:
                primitive = PRIMITIVES[i]
                op = ops.OPS[primitive](C, stride=stride, affine=False, dims=dims)
                if 'pool' in primitive:
                    op = nn.Sequential(op, ops.get_bn_layer(C, affine=False, dims=dims))
                if isinstance(op, ops.Identity) and p > 0:
                    op = nn.Sequential(op, nn.Dropout(self.p))
                self.m_ops.append(op)
                
    def update_p(self):
        for op in self.m_ops:
            if isinstance(op, nn.Sequential):
                if isinstance(op[0], ops.Identity):
                    op[1].p = self.p
                    
    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self.m_ops))


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, switches, p, dims):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.p = p
        if reduction_prev:
            self.preprocess0 = ops.FactorizedReduce(C_prev_prev, C, affine=False, dims=dims)
        else:
            self.preprocess0 = ops.ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False, dims=dims)
        self.preprocess1 = ops.ReLUConvBN(C_prev, C, 1, 1, 0, affine=False, dims=dims)
        self._steps = steps
        self._multiplier = multiplier

        self.cell_ops = nn.ModuleList()
        switch_count = 0
        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, switch=switches[switch_count], p=self.p, dims=dims)
                self.cell_ops.append(op)
                switch_count = switch_count + 1
    
    def update_p(self):
        for op in self.cell_ops:
            op.p = self.p
            op.update_p()

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self.cell_ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C_in, C_pre, num_classes, layers, criterion, config_kwargs, switches_normal=[], switches_reduce=[], step=4, p=0.0, dims=None):
        super(Network, self).__init__()
        self.no_reduction = config_kwargs['no_reduction']
        self.activation = config_kwargs['activation']
        self.remain_shape = config_kwargs['remain_shape']
        self.squeeze = config_kwargs['squeeze']
        self.pool_k = config_kwargs['pool_k']

        self._steps = step
        self._criterion = criterion
        self.p = p
        self.switches_normal = switches_normal
        self.switch_on = switches_normal[0].count(True)
        self.dims = dims
        stem_multiplier = 1
        multiplier = self._steps

        C_curr = stem_multiplier * C_pre
        self.stem = nn.Sequential(
            ops.get_conv_layer(C_in, C_curr, 3, padding=1, bias=False, dims=dims),
            ops.get_bn_layer(C_curr, dims=dims)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C_pre
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if self.no_reduction == False and i in [layers//3, max(3, 2*layers//3)]:
                C_curr *= 2
                reduction = True
                cell = Cell(self._steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, switches_reduce, self.p, dims=dims)
            else:
                reduction = False
                cell = Cell(self._steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, switches_normal, self.p, dims=dims)

            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr

        self.global_pooling = ops.get_pool_layer(type='adaptiveavg', output_size=1, dims=dims)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._network_parameters = [v for _, v in self.named_parameters()]
        self._initialize_alphas()


    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                if self.alphas_reduce.size(1) == 1:
                    weights = F.softmax(self.alphas_reduce, dim=0)
                else:
                    weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                if self.alphas_normal.size(1) == 1:
                    weights = F.softmax(self.alphas_normal, dim=0)
                else:
                    weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)

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
                s1 = self.global_pooling(s1) if self.pool_k > 0 else s1.mean(dim=-1)
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

    def update_p(self):
        for cell in self.cells:
            cell.p = self.p
            cell.update_p()
    
    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target) 

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2+i))
        num_ops = self.switch_on
        self._arch_parameters = []

        self.alphas_normal = nn.Parameter(torch.FloatTensor(1e-3*np.random.randn(k, num_ops)))
        self._arch_parameters.append(self.alphas_normal)
        if not self.no_reduction:
            self.alphas_reduce = nn.Parameter(torch.FloatTensor(1e-3*np.random.randn(k, num_ops)))
            self._arch_parameters.append(self.alphas_reduce)
        else:
            self.alphas_reduce = nn.Parameter(torch.FloatTensor(np.zeros((k, num_ops))))
    
    def arch_parameters(self):
        return self._arch_parameters

    def network_parameters(self):
        return self._network_parameters

