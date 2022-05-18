import dgl
import torch
import torch.nn as nn
from torch.nn import functional


class RGCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, regularizer, n_rels, num_bases):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(
                dgl.nn.RelGraphConv(in_feats, n_hidden, regularizer=regularizer, num_bases=num_bases, num_rels=n_rels))
            for i in range(1, n_layers - 1):
                self.layers.append(
                    dgl.nn.RelGraphConv(n_hidden, n_hidden, regularizer=regularizer,
                                        num_bases=num_bases, num_rels=n_rels))

            self.layers.append(
                dgl.nn.RelGraphConv(n_hidden, n_classes, regularizer=regularizer, num_bases=num_bases, num_rels=n_rels))
        else:
            self.layers.append(
                dgl.nn.RelGraphConv(in_feats, n_classes, regularizer=regularizer, num_bases=num_bases, num_rels=n_rels))

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, g, x):
        h = x
        edge_types = g.edata['type']
        norm = g.edata['norm'].view(-1, 1)
        for i, layer in enumerate(self.layers):
            h = layer(g, h, edge_types, norm)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


class LinkPredictor(nn.Module):
    def __init__(self, rgcn, n_rels, reg_param=0.01):
        """
        Parameters
        ----------
        rgcn : RGCN
            Модель графовой нейронной сети
        n_rels : int

        reg_param : float
            Параметр регуляризации
        """
        super().__init__()
        self.rgcn = rgcn
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(n_rels, self.rgcn.n_classes))
        nn.init.xavier_uniform_(self.w_relation, gain=nn.init.calculate_gain('relu'))

    def forward(self, g, x):
        """
        Parameters
        ----------
        g : dgl.DGLHeteroGraph
            граф кодов ОКВЭД
        x : torch.Tensor
            эмбеддинги описаний
        """
        return functional.dropout(self.rgcn(g, x), p=0.2)

    def calc_score(self, embedding, graph):
        """
        Возвращает DistMult. https://pykeen.readthedocs.io/en/stable/api/pykeen.models.DistMult.html

        Parameters
        ----------
        embedding : torch.Tensor
            эмбеддинг узлов
        graph : dgl.DGLHeteroGraph
            граф кодов ОКВЭД
        """
        # DistMult
        source, target, num_relation = graph.edges(form='all')
        edge_types = graph.edata['type'][num_relation]  # edge type
        s = embedding[source]
        r = self.w_relation[edge_types]
        o = embedding[target]
        score = torch.sum(s * r * o, dim=1)
        return score

    def regularization_loss(self, embedding):
        """
        Возвращает l2 регуляризацию в квадрате

        Parameters
        ----------
        embedding : torch.Tensor
            эмбеддинг узлов
        """
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, embedding, pos_graph, neg_graph):
        """
        Вычисляет полную ошибку, по положительным и отрицательным примерам

        Parameters
        ----------
        embedding : torch.Tensor
            эмбеддинг узлов
        pos_graph : dgl.DGLHeteroGraph
            граф кодов ОКВЭД
        neg_graph : dgl.DGLHeteroGraph
            граф случайно созданных связей между кодами ОКВЭД

        """
        pos_score = self.calc_score(embedding, pos_graph)
        neg_score = self.calc_score(embedding, neg_graph)
        score = torch.cat([pos_score, neg_score])
        label = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]).long()
        predict_loss = functional.binary_cross_entropy_with_logits(score, label.float())

        reg_loss = self.regularization_loss(embedding)
        return predict_loss + self.reg_param * reg_loss
