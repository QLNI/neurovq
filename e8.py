"""Nearest-point decoder for the E8 lattice (Conway & Sloane, SPLAG Ch. 4/20).

E8 = D8 union (D8 + 1/2). To decode a point we find the nearest point in each coset and keep
the closer one. D8 is the set of integer vectors with even coordinate sum; its nearest-point
rule rounds to the nearest integer, then if the coordinate sum is odd, flips the coordinate
whose rounding error was largest.
"""
import torch


def _decode_d8(x):
    f = torch.floor(x + 0.5)
    odd = (f.sum(1) % 2 != 0)
    err = x - f
    j = err.abs().argmax(1)
    s = torch.sign(err[torch.arange(len(x)), j])
    s[s == 0] = 1.0
    g = f.clone()
    g[torch.arange(len(x)), j] += s
    return torch.where(odd[:, None], g, f)


def decode_e8(x):
    """x: (N, 8) -> nearest E8 lattice point (N, 8)."""
    a = _decode_d8(x)
    b = _decode_d8(x - 0.5) + 0.5
    da = ((x - a) ** 2).sum(1)
    db = ((x - b) ** 2).sum(1)
    return torch.where((da <= db)[:, None], a, b)
