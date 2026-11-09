import torch
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch.nn import Sequential, Dropout, Linear, ReLU, BatchNorm1d, Parameter


def MLP(channels, batch_norm=True):
    if batch_norm:
        return Sequential(*[Sequential(Linear(channels[i - 1], channels[i]), ReLU(), BatchNorm1d(channels[i], momentum=0.1))
                            for i in range(1, len(channels))])
    else:
        return Sequential(*[Sequential(Linear(channels[i - 1], channels[i]), ReLU()) for i in range(1, len(channels))])


class EdgeConv(MessagePassing):
    """EdgeConv with message nn([x_i, x_j - x_i]).  nn = MLP([2C, Cout, Cout]) = [Linear, ReLU, BN] -> [Linear, ReLU, BN].
    The first Linear is applied to cat(x_i, x_j - x_i), i.e.  W1 x_i + W2 (x_j - x_i) + b = (W1 - W2) x_i + W2 x_j + b,
    so it is computed once per *vertex* (P = x (W1-W2)^T + b, Q = x W2^T) and gathered per edge, instead of one
    (E x 2C) @ (2C x Cout) matmul over all edges (E ~ 12 V). Mathematically identical to the plain version
    (fast_first_layer=False), only the floating point summation order differs."""
    def __init__(self, in_channels, out_channels, nn, aggr='max', fast_first_layer=True, **kwargs):
        super(EdgeConv, self).__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.fast_first_layer = fast_first_layer

    def forward(self, x, edge_index):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        if not self.fast_first_layer:
            return self.propagate(edge_index, x=x)
        lin = self.nn[0][0]                      # Linear(2C, Cout)
        C = self.in_channels
        w1, w2 = lin.weight[:, :C], lin.weight[:, C:]
        P = torch.addmm(lin.bias, x, (w1 - w2).t())
        Q = x @ w2.t()
        return self.propagate(edge_index, P=P, Q=Q)

    def message(self, x_i=None, x_j=None, P_i=None, Q_j=None):
        if P_i is not None:
            h = self.nn[0][2](torch.relu(P_i + Q_j))     # ReLU, BatchNorm of the first MLP layer
            return self.nn[1](h)                          # second layer: Linear, ReLU, BatchNorm
        return self.nn(torch.cat([x_i, (x_j - x_i)], dim=1))

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.out_channels)
        return aggr_out

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)



class GCUTPL(torch.nn.Module):
    def __init__(self, in_channels, out_channels, aggr='max'):
        super(GCUTPL, self).__init__()
        self.edge_conv_tpl = EdgeConv(in_channels=in_channels, out_channels=out_channels,
                                      nn=MLP([in_channels * 2, out_channels, out_channels]), aggr=aggr,
                                      fast_first_layer=in_channels >= 32)  # raw-coordinate layer (6 channels) stays exact: neighbouring points nearly cancel
        # self.edge_conv_geo = EdgeConv(in_channels=in_channels, out_channels=out_channels // 2,
        #                           nn=MLP([in_channels * 2, out_channels // 2, out_channels // 2]), aggr=aggr)
        self.mlp = MLP([out_channels, out_channels])

    def forward(self, x, tpl_edge_index):
        x_tpl = self.edge_conv_tpl(x, tpl_edge_index)
        x_out = self.mlp(x_tpl)
        return x_out
