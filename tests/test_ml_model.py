import torch

from visionscreen.ml.model import EyeSegNet, N_CLASSES


def test_output_shape_matches_input():
    net = EyeSegNet()
    x = torch.zeros(2, 1, 64, 64)
    y = net(x)
    assert y.shape == (2, N_CLASSES, 64, 64)


def test_handles_non_power_of_two_sizes():
    net = EyeSegNet()
    y = net(torch.zeros(1, 1, 60, 100))
    assert y.shape == (1, N_CLASSES, 60, 100)


def test_parameter_count_is_small():
    # must run in real time on a laptop CPU inside the analyzer loop
    n = sum(p.numel() for p in EyeSegNet().parameters())
    assert n < 600_000, f"model too large: {n}"


def test_gradients_flow():
    net = EyeSegNet()
    out = net(torch.rand(1, 1, 64, 64))
    out.mean().backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
