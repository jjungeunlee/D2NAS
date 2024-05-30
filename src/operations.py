import torch
import torch.nn as nn


OPS = {
  'none'         : lambda C, stride, affine, ks=None, ds=None, ps=None, dims=None: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine, ks=None, ds=None, ps=None, dims=None: get_pool_layer(type='avg', kernel_size=3, stride=stride, padding=1, count_include_pad=False, dims=dims),
  'max_pool_3x3' : lambda C, stride, affine, ks=None, ds=None, ps=None, dims=None: get_pool_layer(type='max', kernel_size=3, stride=stride, padding=1, dims=dims),
  'skip_connect' : lambda C, stride, affine, ks=None, ds=None, ps=None, dims=None: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine, dims=dims),
  'sep_conv_3x3' : lambda C, stride, affine, ks=None, ds=None, ps=None, dims=None: SepConv(C, C, 3, stride, 1,    affine=affine, ks=ks, ds=ds, ps=ps, dims=dims),
  'sep_conv_5x5' : lambda C, stride, affine, ks=None, ds=None, ps=None, dims=None: SepConv(C, C, 5, stride, 2,    affine=affine, ks=ks, ds=ds, ps=ps, dims=dims),
  'sep_conv_7x7' : lambda C, stride, affine, ks=None, ds=None, ps=None, dims=None: SepConv(C, C, 7, stride, 3,    affine=affine, ks=ks, ds=ds, ps=ps, dims=dims),
  'dil_conv_3x3' : lambda C, stride, affine, ks=None, ds=None, ps=None, dims=None: DilConv(C, C, 3, stride, 2, 2, affine=affine, ks=ks, ds=ds, ps=ps, dims=dims),
  'dil_conv_5x5' : lambda C, stride, affine, ks=None, ds=None, ps=None, dims=None: DilConv(C, C, 5, stride, 4, 2, affine=affine, ks=ks, ds=ds, ps=ps, dims=dims),
  'conv_7x1_1x7' : lambda C, stride, affine, ks=None, ds=None, ps=None, dims=None: Conv_nx1_nx1(C, C, 7, stride, 3, affine=affine, dims=dims),
}


def get_conv_layer(input_channels, output_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, dims=None):
  if dims == 2:
    return nn.Conv2d(input_channels, output_channels, 
                     kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
  elif dims == 1:
    return nn.Conv1d(input_channels, output_channels, 
                     kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
  else:
    raise ValueError('Invalid convolution dimension')


def get_bn_layer(output_channels, affine=True, dims=None):
  if dims == 2:
    return nn.BatchNorm2d(output_channels, affine=affine)
  elif dims == 1:
    return nn.BatchNorm1d(output_channels, affine=affine)
  else:
    raise ValueError('Invalid batch normalization dimemsion')


def get_pool_layer(type: str, output_size=1, kernel_size=3, stride=1, padding=0, count_include_pad=False, dims=None):
  if dims == 2:
    if type == 'avg':
      return nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding, count_include_pad=count_include_pad)
    elif type == 'max':
      return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    elif type == 'adaptiveavg':
      return nn.AdaptiveAvgPool2d(output_size)
    else:
      raise ValueError('Invalid pooling type')
  elif dims == 1:
    if type == 'avg':
      return nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding, count_include_pad=count_include_pad)
    elif type == 'max':
      return nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
    elif type == 'adaptiveavg':
      return nn.AdaptiveAvgPool1d(output_size)
    else:
      raise ValueError('Invalid pooling type')
  else:
    raise ValueError('Invalid pooling dimension')


class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, dims=None):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      get_conv_layer(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, dims=dims),
      get_bn_layer(C_out, affine=affine, dims=dims)
    )

  def forward(self, x):
    return self.op(x)


class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True, ks=None, ds=None, ps=None, dims=None):
    super(DilConv, self).__init__()
    ks = ks if ks is not None else [kernel_size]
    ds = ds if ds is not None else [dilation]
    ps = ps if ps is not None else [padding]
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      get_conv_layer(C_in, C_in,  kernel_size=ks.pop(0), stride=stride, padding=ps.pop(0), dilation=ds.pop(0), groups=C_in, dims=dims),
      get_conv_layer(C_in, C_out, kernel_size=1, padding=0, dims=dims),
      get_bn_layer(C_out, affine=affine, dims=dims),
    )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, ks=None, ds=None, ps=None, dims=None):
    super(SepConv, self).__init__()
    ks = ks if ks is not None else [kernel_size, kernel_size]
    ds = ds if ds is not None else [1, 1]
    ps = ps if ps is not None else [padding, padding]
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      get_conv_layer(C_in, C_in, kernel_size=ks.pop(0), stride=stride, padding=ps.pop(0), dilation=ds.pop(0), groups=C_in, dims=dims),
      get_conv_layer(C_in, C_in, kernel_size=1, padding=0, dims=dims),
      get_bn_layer(C_in, affine=affine, dims=dims),
      nn.ReLU(inplace=False),
      get_conv_layer(C_in, C_in, kernel_size=ks.pop(0), stride=1, padding=ps.pop(0), dilation=ds.pop(0), groups=C_in, dims=dims),
      get_conv_layer(C_in, C_out, kernel_size=1, padding=0, dims=dims),
      get_bn_layer(C_out, affine=affine, dims=dims),
    )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    input_shape = list(x.size())
    for i in range(2, len(input_shape)): # for width, height
      input_shape[i] += input_shape[i] % self.stride #to match the output of other op
      input_shape[i] //= self.stride
    if x.is_cuda:
      with torch.cuda.device(x.get_device()):
        padding = torch.cuda.FloatTensor(*input_shape).fill_(0)
    else:
      padding = torch.FloatTensor(*input_shape).fill_(0)
    return padding


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True, dims=None):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = get_conv_layer(C_in, C_out // 2, kernel_size=1, stride=2, padding=0, dims=dims)
    self.conv_2 = get_conv_layer(C_in, C_out // 2, kernel_size=1, stride=2, padding=0, dims=dims) 
    self.bn = get_bn_layer(C_out, affine=affine, dims=dims)

  def forward(self, x):
    x = self.relu(x)
    x1 = self.conv_1(x)
    if x.dim() == 4:
      x2 = self.conv_2(x[:,:,1:,1:])
    else:
      x2 = self.conv_2(x[:,:,1:])
    out = torch.cat([x1, x2 if x1.size() == x2.size() else x1], dim=1)
    out = self.bn(out)
    return out


class Conv_nx1_nx1(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, dims=None):
    super(Conv_nx1_nx1, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      get_conv_layer(C_in, C_in , kernel_size=(1,kernel_size), stride=(1,stride), padding=(0,padding), dims=dims),
      get_conv_layer(C_in, C_out, kernel_size=(kernel_size,1), stride=(stride,1), padding=(padding,0), dims=dims),
      get_bn_layer(C_out, affine=affine, dims=dims),
    )

  def forward(self, x):
    return self.op(x)