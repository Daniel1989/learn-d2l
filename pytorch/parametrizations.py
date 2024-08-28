import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

def symmetric(X):
    return X.triu() + X.triu(1).transpose(-1, -2)

X = torch.rand(3, 3)
A = symmetric(X)
assert torch.allclose(A, A.T)  # A is symmetric
print(A)


class LinearSymmetric(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(n_features, n_features))

    def forward(self, x):
        A = symmetric(self.weight)
        return x @ A

layer = LinearSymmetric(3)
out = layer(torch.rand(8, 3))