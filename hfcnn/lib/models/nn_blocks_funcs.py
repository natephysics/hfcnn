import math
import logging
from collections import abc
from typing import Union, Tuple
from itertools import repeat

import torch
import torch.nn as nn


######
#       Funcs
######

def fit_input_dims(network_structure, input_dims):
    # deprecated
    # fit input dims to network structure (conv/lin) no of inputs
    # check that input_dims are correct for heads in network_structure
    if isinstance(input_dims, (int, tuple)):  # cnn/linear input
        assert isinstance(network_structure, list) or network_structure.keys() == [0], \
            'Single input needs single connector in nn'

    elif isinstance(input_dims, dict):
        input_keys = [k for k in network_structure.keys()]
        assert all(isinstance(k, int) for k in input_keys), 'Only integer keys as network_structure.keys()'
        assert set(input_keys) == set(input_dims.keys()), f'Found different keys as inputs to network structure:' \
                                                          f'{set(input_dims).symmetric_difference(set(input_dims.keys()))}'
        for route_key, route_dim in input_dims.items():
            if isinstance(route_dim, int):  # linear
                assert network_structure[route_key][0] == 'linear', f'Input at {route_key} needs linear layer but ' \
                                                                    f'{network_structure[route_key][0]} provided'
            elif isinstance(route_dim, tuple):  # cnn
                assert len(route_dim) == 3, f'Input to Cnn format: Channel, Height, Width but is {route_dim}'
                assert network_structure[route_key][0] == 'convolutional', \
                    f'Input at {route_key} needs convolutional layer but ' \
                    f'{network_structure[route_key][0]} provided'


def pair(x):
    if isinstance(x, abc.Iterable):
        return x
    return tuple(repeat(x, 2))


def init_xavier_uniform(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        torch.nn.init.xavier_uniform_(m.weight)


def init_orthogonal(m, gain=1):
    if type(m) in [nn.Linear, nn.Conv2d]:
        torch.nn.init.orthogonal_(m.weight, gain=gain)


def init_bias(m, constant=0):
    if type(m) in [nn.Linear, nn.Conv2d]:
        torch.nn.init.constant_(m.bias, val=constant)


def calc_out_hw(hw_in:  int, padding: int, dilation: int, kernel_size: int, stride: int
                ) -> int:
    return math.floor(((hw_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)


def calc_avgpool_out_hw(hw_in: int, padding: int, kernel_size: int, stride: int):
    return math.floor(((hw_in + 2 * padding - kernel_size) / stride) + 1)


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Clamp(torch.nn.Module):
    def __init__(self, min, max):
        super(Clamp, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.sigmoid(x) * (self.max - self.min) + self.min


######
#       Blocks
######

# class CompositeModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.sequentials = []
#     def add_seq(self, seq):
#        self.sequentials
#     def


class LinearLayer(nn.Module):
    def __init__(self, in_dim: Union[int, Tuple[int, int, int]], **kwargs):
        super().__init__()
        out_features = kwargs.pop('out', 256)
        bias = kwargs.pop('bias', False)
        if len(kwargs):
            logging.warning(f'Linear Unknown kwargs: {kwargs}')
        self.lin = nn.Sequential()
        if isinstance(in_dim, tuple):
            if len(in_dim) == 3:
                logging.warning('Got tuple as in dim, implicit flattening input')
                self.lin.add_module('flat', Flatten())
                in_dim = in_dim[0]*in_dim[1]*in_dim[2]
            elif len(in_dim) == 1:
                in_dim = in_dim[0]
            else:
                raise ValueError(f'{in_dim} input dimension type not correct - Union[int, Tuple[int, int, int]]')
        self.lin.add_module('lin',nn.Linear(in_dim,
                                            out_features=out_features,
                                            bias=bias))
        self.out_dim = out_features
        
    def forward(self, x):
        for op in self.lin:
            x = op(x)
        return x


class ConvLayer2D(nn.Module):
    # Todo specify output size and infer fitting kernel_size
    def __init__(self, in_dim: Tuple[int, int, int], **kwargs):
        super().__init__()
        in_channels = in_dim[0]
        out_channels = kwargs.pop('out', 4)
        kernel_size = kwargs.pop('kernel_size', 3)
        stride = kwargs.pop('stride', 1)
        padding = kwargs.pop('padding', 0)
        dilation = kwargs.pop('dilation', 1)
        bias = kwargs.pop('bias', True)
        if len(kwargs):
            logging.warning(f'Conv2d Unknown kwargs: {kwargs}')
        padding_mode = kwargs.pop('padding_mode', 'zeros')
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=bias,
                              padding_mode=padding_mode)
        self.out_dim = (out_channels, 
                        calc_out_hw(in_dim[1], self.conv.padding[0], self.conv.dilation[0],
                                    self.conv.kernel_size[0], self.conv.stride[0]),
                        calc_out_hw(in_dim[2], self.conv.padding[1], self.conv.dilation[1],
                                    self.conv.kernel_size[1], self.conv.stride[1]))

    def forward(self, x):
        return self.conv(x)


class MergeLayer(nn.Module):
    def __init__(self, in_routes: list, device):
        assert isinstance(in_routes, list), f'{in_routes} key is route to be merged and value is dim'
        super().__init__()
        self.out_dim = None
        self.device = device
        self._parse_ins(in_routes)
        # parse type of in_routes and check compatibility
        # (cnn->h1=h2,w1=w2, sum(channels); linear+cnn->Flatten; linear->sum(ins))

    def op(self, x):
        if len(x.shape) >= 3:
            return x.reshape(x.size(0), -1).squeeze(0)
        else:
            return x.squeeze(0)

    def forward(self, x: list):
        ret = torch.FloatTensor([]).to(self.device)

        for v in x:
            insi = self.op(v)
            ret = torch.cat((ret, insi), dim=1)
        return ret

    def _parse_ins(self, in_routes: list):
        if all(isinstance(x, tuple) for x in in_routes):  # merge cnn's
            assert all(in_routes[i][1] == in_routes[i+1][1] and in_routes[i][2] == in_routes[i+1][2]
                       for i in range(len(in_routes)-1)), \
                f'height and width have to be the same in merged cnns: {in_routes}'
            self.out_dim = (sum(i[0] for i in in_routes), in_routes[0][1], in_routes[0][2])  # C, H, W
        elif all(isinstance(x, (int, tuple)) for x in in_routes):  # mixed
            logging.warning(f'Mixed input for merge, flattening everything at {in_routes}')
            self.out_dim = sum(i if isinstance(i, int) else i[0]*i[1]*i[2] for i in in_routes)
        elif all(isinstance(x, int) for x in in_routes):   # linear
            self.out_dim = sum(i for i in in_routes)
        else:
            raise NotImplementedError(f'Unknown type in merge layer {in_routes}')


class SplitLayer(nn.Module):
    def __init__(self, split: float):
        super().__init__()
        assert 1 > split > 0, f'split has to be in [0,1]: {split}'
        self.p = split

    def forward(self, x):
        return torch.split(x, int(x.shape[1] * self.p), dim=self.dim)


class MaxPool(nn.Module):
    def __init__(self, in_dim: Union[int, Tuple[int, int, int]], **kwargs):
        super().__init__()
        adaptive = kwargs.pop('adaptive', False)
        kernel_size = kwargs.pop('kernel_size', 3)
        stride = kwargs.pop('stride', 1)
        padding = kwargs.pop('padding', 0)
        dilation = kwargs.pop('dilation', 1)

        kernel_size = pair(kernel_size)
        stride = pair(stride)
        padding = pair(padding)
        dilation = pair(dilation)

        self.op = None
        self.out_dim = None
        linear = False

        if isinstance(in_dim, int):
            linear = True
        elif isinstance(in_dim, tuple):
            assert len(in_dim) == 3, f'Input dim has to be of length 3 (c,h,w)'
        else:
            raise ValueError(f'Wrong input type for in_dim: {in_dim}')

        if adaptive:
            output_size = kwargs.pop('output_size', None)
            if output_size is None:
                if linear:
                    output_size = in_dim - in_dim // 5
                    self.op = nn.AdaptiveMaxPool1d(output_size=output_size)
                    self.out_dim = output_size
                else:
                    output_size = (in_dim[1] - 3, in_dim[1] - 3)
                    self.op = nn.AdaptiveMaxPool2d(output_size=output_size)
                    self.out_dim = (in_dim[0], output_size[0], output_size[1])
                logging.warning(f'No output size for adaptive pooling given, {output_size} chosen')

            else:
                if linear:
                    self.op = nn.AdaptiveMaxPool1d(output_size=output_size)
                else:
                    self.op = nn.AdaptiveMaxPool2d(output_size=output_size)
                self.out_dim = output_size
        else:
            if linear:
                self.op = nn.MaxPool1d(kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding, dilation=dilation)
                self.out_dim = (in_dim[0],
                                calc_out_hw(in_dim[1], self.op.padding[0], self.op.dilation[0],
                                            self.op.kernel_size[0], self.op.stride[0]),
                                calc_out_hw(in_dim[2], self.op.padding[1], self.op.dilation[1],
                                            self.op.kernel_size[1], self.op.stride[1])
                                )
            else:
                self.op = nn.MaxPool2d(kernel_size=kernel_size,
                                       stride=stride, padding=padding,
                                       dilation=dilation)
                self.out_dim = (in_dim[0],
                                calc_out_hw(in_dim[1], self.op.padding[0], self.op.dilation[0],
                                            self.op.kernel_size[0], self.op.stride[0]),
                                calc_out_hw(in_dim[2], self.op.padding[1], self.op.dilation[1],
                                            self.op.kernel_size[1], self.op.stride[1])
                                )

    def forward(self, x):
        return self.op(x)


class AvgPool(nn.Module):
    def __init__(self, in_dim: Union[int, Tuple[int, int, int]], **kwargs):
        super().__init__()
        adaptive = kwargs.pop('adaptive', False)
        kernel_size = kwargs.pop('kernel_size', 3)
        stride = kwargs.pop('stride', 1)
        padding = kwargs.pop('padding', 0)

        kernel_size = pair(kernel_size)
        stride = pair(stride)
        padding = pair(padding)

        self.op = None
        self.out_dim = None
        linear = False

        if isinstance(in_dim, int):
            linear = True
        elif isinstance(in_dim, tuple):
            assert len(in_dim) == 3, f'Input dim has to be of length 3 (c,h,w)'
        else:
            raise ValueError(f'Wrong input type for in_dim: {in_dim}')

        if adaptive:
            output_size = kwargs.pop('output_size', None)
            if output_size is None:
                if linear:
                    output_size = in_dim - in_dim // 5
                    self.op = nn.AdaptiveAvgPool1d(output_size=output_size)
                    self.out_dim = output_size
                else:
                    output_size = (in_dim[1] - 3, in_dim[1] - 3)
                    self.op = nn.AdaptiveAvgPool2d(output_size=output_size)
                    self.out_dim = (in_dim[0], output_size[0], output_size[1])
                logging.warning(f'No output size for adaptive pooling given, {output_size} chosen')

            else:
                if linear:
                    self.op = nn.AdaptiveAvgPool1d(output_size=output_size)
                else:
                    self.op = nn.AdaptiveAvgPool2d(output_size=output_size)
                self.out_dim = output_size
        else:
            if linear:
                self.op = nn.AvgPool1d(kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding)
                self.out_dim = (in_dim[0],
                                calc_avgpool_out_hw(in_dim[1], self.op.padding[0],
                                                    self.op.kernel_size[0], self.op.stride[0]),
                                calc_avgpool_out_hw(in_dim[2], self.op.padding[1],
                                                    self.op.kernel_size[1], self.op.stride[1])
                                )
            else:
                self.op = nn.AvgPool2d(kernel_size=kernel_size,
                                       stride=stride, padding=padding)
                self.out_dim = (in_dim[0],
                                calc_avgpool_out_hw(in_dim[1], self.op.padding[0],
                                                    self.op.kernel_size[0], self.op.stride[0]),
                                calc_avgpool_out_hw(in_dim[2], self.op.padding[1],
                                                    self.op.kernel_size[1], self.op.stride[1])
                                )

    def forward(self, x):
        return self.op(x)


if __name__ == '__main__':
    # ml = MergeLayer([(1,23,113), 199])
    # print(ml([torch.zeros(1,23,113), torch.zeros(199)]).shape)

    # ml = MergeLayer([(1, 23, 113), (1, 23, 113)])
    # print(ml([torch.zeros(1, 23, 113), torch.zeros(1, 23, 113)]).shape)

    # ml = MergeLayer([113, 287])
    # print(ml([torch.zeros(113), torch.zeros(287)]).shape)

    pass
