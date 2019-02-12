import torch
import torch.nn as nn

from numpy.testing import assert_almost_equal


def loss_fn(pred, gt):
    return (pred - gt) ** 2


def test_partial_gradient():
    y = torch.tensor([0.], requires_grad=False)
    W1 = torch.arange(9, requires_grad=True, dtype=torch.float).view(3, 3)
    W1.retain_grad()
    X = torch.tensor([[1, 0, -1], [2, 0, -2], [1, -1, -1]], dtype=torch.float, requires_grad=False)
    H = W1 * X
    H.retain_grad()
    W21 = torch.arange(9, requires_grad=True, dtype=torch.float).view(3, 3)
    W22 = torch.arange(9, requires_grad=True, dtype=torch.float).view(3, 3) * -1
    S1 = (W21 * H).sum()
    S2 = (W22 * H).sum()
    l1 = loss_fn(S1, y)
    l2 = loss_fn(S2, y)
    w1 = torch.tensor([.3], requires_grad=True)
    w2 = torch.tensor([.7], requires_grad=True)
    L1 = w1 * l1
    L2 = w2 * l2
    L = L1 + L2
    L.backward(retain_graph=True)
    G1R = torch.autograd.grad(L1, W1, retain_graph=True, create_graph=True)
    G2R = torch.autograd.grad(L2, W1, retain_graph=True, create_graph=True)
    assert_almost_equal((G1R[0] + G2R[0]).detach().numpy(), W1.grad.numpy())
