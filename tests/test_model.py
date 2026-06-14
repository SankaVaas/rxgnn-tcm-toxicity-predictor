"""Smoke tests for RxGNN."""
import torch, pytest


@pytest.fixture
def tiny():
    N, E = 20, 40
    return (torch.randn(N, 16), torch.randint(0, N, (2, E)),
            torch.randint(0, 5, (E,)), torch.randint(0, N, (8, 2)), torch.randn(8, 5))


def test_forward_shapes(tiny):
    from rxgnn import RxGNN
    m = RxGNN(in_dim=16, hidden=32, n_rel=5, n_layers=2, dropout=0.0)
    tox, metab = m(*tiny)
    assert tox.shape == (8,) and metab.shape == (20,)


def test_no_nan(tiny):
    from rxgnn import RxGNN
    m = RxGNN(in_dim=16, hidden=32, n_rel=5, n_layers=2, dropout=0.0)
    tox, metab = m(*tiny)
    assert not torch.isnan(tox).any() and not torch.isnan(metab).any()