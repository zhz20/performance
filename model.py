# -*- encoding = utf-8 -*-
"""
@description: hypergraph neural network
@date: 2024/1/17
@File : model.py
@Software : PyCharm
"""


import dhg
import torch
import torch.nn as nn
from dhg.structure.graphs import Graph



class HGNNConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            drop_rate: float = 0.1,
    ):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, hg: dhg.Hypergraph) -> torch.Tensor:
        X = self.theta(X)
        X_hg = hg.smoothing_with_HGNN(X)  # HGNN
        X_ = self.drop(self.act(X_hg))
        return X_


class HGNN(nn.Module):
    def __init__(self, in_channels, h_channels, out_channels):
        super(HGNN, self).__init__()
        self.conv1 = HGNNConv(in_channels, h_channels)
        self.conv2 = HGNNConv(h_channels, out_channels)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, 1),
            nn.Sigmoid()
        )

    def encode(self, X, hg):
        x = self.conv1(X, hg).relu()
        return self.conv2(x, hg)

    def forward(self, X, hg, link):
        # encoder
        x = self.encode(X, hg)

        out = []

        for edge in link.e[0]:
            # mean pooling
            edge_x = torch.cat([x[i].unsqueeze(0) for i in edge], dim=0).mean(dim=0)
            out.append(edge_x)

        out = torch.stack(out)  # Convert to Tensor
        out = self.fc(out)  # Prediction
        return out


class HGNNPLUSConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            drop_rate: float = 0.1,
    ):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, hg: dhg.Hypergraph) -> torch.Tensor:
        X = self.theta(X)
        Y = hg.v2e(X, aggr="mean")    # HGNN+
        X_hg = hg.e2v(Y, aggr="mean")
        X_ = self.drop(self.act(X_hg))
        return X_


class HGNNPLUS(nn.Module):
    def __init__(self, in_channels, h_channels, out_channels):
        super(HGNNPLUS, self).__init__()
        self.conv1 = HGNNPLUSConv(in_channels, h_channels)
        self.conv2 = HGNNPLUSConv(h_channels, out_channels)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, 1),
            nn.Sigmoid()
        )

    def encode(self, X, hg):
        x = self.conv1(X, hg).relu()
        return self.conv2(x, hg)

    def forward(self, X, hg, link):
        # encoder
        x = self.encode(X, hg)

        out = []

        for edge in link.e[0]:
            # mean pooling
            edge_x = torch.cat([x[i].unsqueeze(0) for i in edge], dim=0).mean(dim=0)
            out.append(edge_x)

        out = torch.stack(out)  # Convert to Tensor
        out = self.fc(out)  # Prediction
        return out