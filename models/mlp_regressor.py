import numpy as np
import torch
from torch import nn
from torch import optim


class MLPRegressorTorch(nn.Module):
    def __init__(self, input_dim: int, layer_dims: tuple[int, ...] = (100,), activation=nn.ReLU()):
        super(MLPRegressorTorch, self).__init__()
        self.layers = [nn.Linear(input_dim, layer_dims[0]), activation]
        for i in range(1, len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            self.layers.append(activation)
        else:
            self.layers.append(nn.Linear(layer_dims[-1], 1))
            self.layers.append(activation)
        self.layers = nn.Sequential(*self.layers)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.layers(x)

    def r2_score(self, x, y):
        from sklearn.metrics import r2_score
        y_pred = self.forward(x).flatten()
        return r2_score(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())

    def get_loss(self, x, y):
        x = self.forward(x).flatten()
        return self.criterion(y, x)

    def fit(self, x, y, val: list = None, epochs: int = 1, verbose=False):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        show_epochs = list(np.unique(np.geomspace(1, epochs, 15, dtype=int)))
        last_improvement = 0
        best_val_loss = 10 ** 10
        best_state = self.state_dict()

        for i in range(epochs):
            loss = self.get_loss(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Валидация
            if val is not None:
                with torch.no_grad():
                    val_loss = self.get_loss(val[0], val[1]).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    last_improvement = 0
                    best_state = self.state_dict()
                else:
                    last_improvement += 1

            # Печать
            if verbose:
                if i in show_epochs + [epochs - 1]:
                    with torch.no_grad():
                        if val is not None:
                            val_loss = self.get_loss(val[0], val[1]).item()
                            print(f'epoch {i + 1: 6d}  :  loss : {loss.item(): .5f} | val loss : {val_loss: .5f}')
                        else:
                            print(f'epoch {i + 1: 6d}  :  loss : {loss.item(): .5f}')
        else:
            if last_improvement > 0:
                self.load_state_dict(best_state)
