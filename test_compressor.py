import numpy as np
import pytest
import torch
from torch import Tensor

#from clarity.enhancer.compressor import Compressor
from compressor import Compressor

DEFAULT_FS = 44100


def test_compressor_set_attack():
    """Test that the attack time is set correctly."""
    c = Compressor()
    c.set_attack(1000)

    assert c.attack == 1.0 / DEFAULT_FS


def test_compressor_set_attack_error():
    """Test that the attack time raises divide by zero error if set to 0"""
    c = Compressor()
    with pytest.raises(ZeroDivisionError):
        c.set_attack(0)


def test_compressor_set_release():
    """Test that the release time is set correctly."""
    c = Compressor()
    c.set_release(1000)

    assert c.release == 1.0 / DEFAULT_FS


def test_compressor_set_release_error():
    """Test that the release time raises divide by zero error if set to 0."""
    c = Compressor()
    with pytest.raises(ZeroDivisionError):
        c.set_release(0)


def test_compressor_process():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    c = Compressor()
    signal = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float).to(device)
    output, rms, comp_ratios = c.process(signal)

    assert len(output) == len(signal)
    assert torch.all(rms >= 0.0)
    assert torch.sum(rms) == pytest.approx(
        0.9799197751960967, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert len(comp_ratios) == len(signal)
