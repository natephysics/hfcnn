"""
Todo
    graph structure for description
    keras/tf backend + comparison to torch
    yaml / json as input

    Possible better to make this class based in
    split class, combine class and throughput class which
    get composed by the network structure
    also reverse searching easier (find input space for given input dim)
"""

import copy
import logging
from typing import Union, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torchinfo

from hfcnn.lib.models.nn_blocks_funcs import \
    LinearLayer, ConvLayer2D, MergeLayer, SplitLayer, \
    MaxPool, AvgPool


class CustomSequential(nn.Sequential):
    def forward(self, **inputs):
        out = {}
        for k, b in self.named_children():
            if not set(map(int, inputs.keys())).issubset(set(out.keys())):
                out[int(k)] = b(inputs[k])
                k = int(k)
                #del inputs[k]
            else:
                # if k.startswwith('merge_')  # todo more finegrained logic (merge, split)
                k = tuple(map(int, k.replace('(','').replace(')','').split(', ')))
                out[k] = b([out[i] for i in k])

        return out[k]   # return last element of sequential


def parse_in_dim_network_structure(in_dim, network_structure):
    # in_dim can be dict for multiple inputs
    # tuple(int, int) for conv2d
    # tuple(int,) or in_dim for linear

    assert isinstance(network_structure, dict), 'Only dict as network structure'

    assert isinstance(in_dim, (tuple, dict, int)), \
        f'Input dimension has to be tuple/dict but is: {in_dim}'

    if isinstance(in_dim, (int, tuple)):  # cnn/linear input
        assert list(network_structure.keys()) == [0], \
            f'Single input value needs single connector in nn, ' \
            f'but there are multiple inputs: {network_structure.keys()}'
        if isinstance(in_dim, int):
            in_dim = (in_dim,)

    def check_connection(block, in_d):
        if len(in_d) == 1:
            # linear, 1d
            assert block[0][0] == 'linear', \
                    f'{in_d} as input dimension requires ' \
                    f'linear layer but is {block}'
        elif len(in_d) == 3:
            # conv, 2d
            assert block[0][0] == 'convolutional', \
                f'{in_d} as input dimension requires ' \
                f'convolutional layer but is {block}'
        else:
            raise ValueError(f'tuple in_dim can either be 3d (C,H,W) or 1d, not {in_d}')

    if isinstance(in_dim, tuple):
        # simple one layer in, one layer out network
        check_connection(network_structure[0], in_dim)

    if isinstance(in_dim, dict):
        # multiple layer/inputs in, one layer out network
        input_keys = [k for k in network_structure.keys()]
        # assert all(isinstance(k, int) for k in input_keys), \
        #     'Only integer keys as network_structure.keys()'
        # if not set(input_keys) == set(in_dim.keys()):
        #     diff = set(input_keys).symmetric_difference(set(in_dim.keys()))
        #     raise ValueError(f'Found different keys as inputs to network structure:'
        #                f'{diff}. In_dims={in_dim}, input_keys={input_keys}')
        # for k in input_keys:
        #     check_connection(network_structure[k], in_dim[k])


class NeuralNetwork(nn.Module):
    def __init__(self, in_dim: Union[int, Tuple[int, int, int], dict],
                 network_structure: Union[list, dict]):

        network_structure = copy.deepcopy(network_structure)

        parse_in_dim_network_structure(in_dim, network_structure)

        super(NeuralNetwork, self).__init__()

        # Todo redefine as nn.Sequentials ordered by route keys
        #   then at the correct place (route keys) concat output of sequentials perform last forward pass
        #   make possible to split computational graph
        #   keep track of different blocks
        #   add split/unsplit keywords
        #   implement compositions of this class for bigger nets, residual paths?

        self.operators = CustomSequential()
        self.dummy = None

        if isinstance(in_dim, int) or isinstance(in_dim, tuple):
            if isinstance(network_structure, list): # single block
                network_structure = OrderedDict({0: network_structure})
            if isinstance(in_dim, tuple):
                if len(in_dim) == 2:
                    logging.warning(f'appending dimension to {in_dim} (input is 1 channel)')
                    in_dim = (1, *in_dim)
            self.dummy = self._get_dummy(in_dim)

            self.in_dim = {0: in_dim}
        # check that in_dim fits network structure if multiple inputs
        elif isinstance(in_dim, dict):
            assert isinstance(network_structure, dict), \
                'multiple inputs require {input_idx: structure} as network_structure dict'
            self.dummy = {}
            # order dict and init dummy inputs
            network_s = OrderedDict()
            for key, dimension in in_dim.items():
                if isinstance(key, int) and isinstance(dimension, tuple):
                    if len(dimension) == 2:
                        logging.warning(f'appending dimension to {dimension} (input is 1 channel)')
                        dimension = (1, *dimension)
                        in_dim[key] = dimension
                try:
                    network_s[key] = network_structure[key]
                    del network_structure[key]
                except KeyError:
                    raise KeyError(f"{key} not in given network structure, check state_dim")
                self.dummy[str(key)] = self._get_dummy(dimension)
            for key, val in network_structure.items():
                network_s[key] = val
            network_structure = copy.deepcopy(network_s)
            self.in_dim = in_dim

        else:
            raise ValueError(f'{in_dim} not viable input dimensions')

        self._build_net(network_structure, self.in_dim)

    @staticmethod
    def _get_dummy(in_dim: Union[int, Tuple[int, int, int]]):
        if isinstance(in_dim, int):
            dummy = torch.zeros(in_dim).unsqueeze(0)
        # elif isinstance(in_dim, tuple) and len(in_dim) == 1:
        #     dummy = torch.zeros(in_dim).unsqueeze(0)
        elif isinstance(in_dim, tuple):
            if len(in_dim) == 3:
                dummy = torch.zeros(in_dim).unsqueeze(0)  # add batch dim
            # elif len(in_dim) == 2:
            #     dummy = torch.zeros(in_dim).unsqueeze(0).unsqueeze(0)
            elif len(in_dim) == 1:
                dummy = torch.zeros(in_dim).unsqueeze(0)
            else:
                raise ValueError(f'Unrecognized in_dim tuple: {in_dim}')
        else:
            raise ValueError(f'Input dim {in_dim} has to have shape int or tuple')
        return dummy

    def _build_net(self, network_structure: dict, in_dim):
        # todo _build_block should return block as list, then add to OrderedDict
        dims = copy.deepcopy(in_dim)
        for route, block in network_structure.items():
            blocklist = []
            if isinstance(route, int):  # input
                route_dim = dims.pop(route)
                out_dim, blocklist = self._build_block(route_dim, block)
                dims.update({route: out_dim})
                self.operators.add_module(str(route), nn.Sequential(*blocklist))
            elif isinstance(route, tuple):  # merge
                route_dim = [dims[i] for i in route]
                merge_op = MergeLayer(route_dim)
                out_dim, block = self._build_block(merge_op.out_dim, block)
                blocklist.append(merge_op)
                blocklist.extend(block)
                dims.update({route: out_dim})
                self.operators.add_module(str(route), nn.Sequential(*blocklist))


            else:
                raise ValueError(f'Unknown route shape: {route}')
            #
            # todo split/unsplit keywords
            #       sanity check the resulting network structure

    def _build_block(self, in_dim: Union[int, Tuple[int, int, int]],
                     block_structure: list) -> Tuple[Union[int, Tuple[int, int, int]], list]:
        linear = False
        block = []
        for (layer, params) in block_structure:
            if layer == 'convolutional':
                assert not linear, 'Can only use conv2d before linear layers'
                layer = ConvLayer2D(in_dim, **params)
                block.append(layer)
                in_dim = layer.out_dim
            elif layer == 'linear':
                layer = LinearLayer(in_dim, **params)
                block.append(layer)
                in_dim = layer.out_dim
                linear = True
            elif layer == 'relu':
                # todo explicit None or warning needed if params given?
                block.append(nn.ReLU())
            elif layer == 'leaky_relu':
                if params is None:
                    params = {}
                slope = params.get('slope', .01)
                block.append(nn.LeakyReLU(slope))
            elif layer == 'selu':
                block.append(nn.SELU())
            elif layer == 'tanh':
                block.append(nn.Tanh())
            elif layer == 'gelu':
                block.append(nn.GELU())
            elif layer == 'dropout':
                if params is None:
                    params = {}
                p = params.get('p', 5)
                block.append(nn.Dropout(p=p))
            # BatchNorm + Multigpu training (dataparallel) needs extra care
            elif layer == 'batchnorm':
                if params is None:
                    params = {}
                momentum = params.get('momentum', .1)
                if linear:
                    block.append(nn.BatchNorm1d(in_dim, momentum=momentum))
                else:
                    block.append(nn.BatchNorm2d(in_dim[0], momentum=momentum))
            elif layer == 'maxpool':
                layer = MaxPool(in_dim, **params)
                block.append(layer)
                in_dim = layer.out_dim
            elif layer == 'avgpool':
                layer = AvgPool(in_dim, **params)
                block.append(layer)
                in_dim = layer.out_dim
            else:
                raise NotImplementedError(f'{layer} not known')

        return in_dim, block


    def plot(self):
        self.eval()
        # dummy = {}
        # for k, v in self.dummy.items():
        #     dummy[int(k)] = v
        # torchinfo.summary(self, input_data=self.dummy)
        if isinstance(self.dummy, dict):
            torchinfo.summary(self, input_size=None, **self.dummy)
        else:
            torchinfo.summary(self, input_data=self.dummy)

    def forward(self, inputs, **kwargs):
        # if kwargs ...
        # a = self.operators._modules.values()
        # b = [c for c in self.operators.named_children()]
        # print(self)
        # for key, block in self.operators.named_children():
        #     if key.startswith('merge_'):
        #         kwargs[key] = block
        #         continue
        #     kwargs[key] = block(kwargs[key])


        # out = {}
        # for route, seq in self.operators.named_children():
        #     if not set(map(int, inputs.keys())).issubset(set(out.keys())):
        #         out[int(route)] = seq(inputs[route])
        # todo put customsequential here

        if isinstance(inputs, dict):
            ret = self.operators.forward(**inputs)
        else:
            ret = self.operators.forward(**{'0': inputs})
        return ret

    def get_weights(self):
        raise NotImplementedError

    def get_activations(self):
        raise NotImplementedError


if __name__ == '__main__':
    def test_cnn_lin_to_lin():
        structure = \
            {0:
            [
                         ['convolutional', {'out': 45,"kernel_size": [6, 6]}],
                         ['relu', None],
                         ['convolutional', {"kernel_size": [6, 6]}],
                         ['relu', None],
            ],

            1:
                     [
                     ['linear', {'out': 512}],
                     ['relu', None],
                     ['linear', {'out': 256}],
                     ['relu', None]
                     ],
            [0, 1]:
                    [
                     ['linear', {'out': 64}],
                     ['relu', None]
                    ]
        }
        in_dimension = {0: (1, 29, 113),
                        1: 7}
        neuraln = NeuralNetwork(in_dim=in_dimension,
                                network_structure=structure)

        neuraln.plot()

    def test_lin():
        structure = [
                     ('linear', {'out': 512}),
                     ('relu', None),
                     ('linear', {'out': 256}),
                     ('relu', None)
                     ]
        neuraln = NeuralNetwork(in_dim=23,
                                network_structure=structure)

        neuraln.plot()

    def test_cnn_lin():
        structure = [
                         ('convolutional', {"kernel_size": (6, 6)}),
                         ('relu', None),
                         ('maxpool', {}),
                         ('convolutional', {"kernel_size": (6, 6)}),
                         ('batchnorm', {}),
                         ('relu', None),
                         ('maxpool', {}),
                        ('linear', {'out': 512}),
                        ('relu', None),
                        ('linear', {'out': 512}),
                        ('relu', None),
                        ('linear', {'out': 256}),
                        ('relu', None)
                    ]

        neuraln = NeuralNetwork(in_dim=(1,29,113),
                                network_structure=structure)
        # print(neuraln)
        # print(neuraln.dummy)
        # abc = neuraln.forward(neuraln.dummy)
        # print('ret:', abc.shape)
        # print(neuraln.dummy)
        neuraln.plot()

    # test_lin()
    # test_cnn_lin()
    test_cnn_lin_to_lin()