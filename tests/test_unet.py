"""Tests for basic UNet forward execution and model loading."""

import torch

from afmlevel.unet import _MODEL_CACHE, load_unet_model


def test_unet_forward_shape(tiny_unet_cpu):
    """Ensure a forward pass preserves the expected input/output shape."""
    x = torch.zeros(1, 1, 256, 256)
    with torch.no_grad():
        out = tiny_unet_cpu(x)
    assert out.shape == (1, 1, 256, 256)


def test_unet_load_from_local_pth(tmp_path, tiny_unet_cpu):
    """Verify a UNet can be saved to disk and reloaded for inference."""
    # Avoid cache pollution between tests
    _MODEL_CACHE.clear()

    pth_file = tmp_path / "test_model.pth"
    torch.save(tiny_unet_cpu.state_dict(), pth_file)

    config = {
        "filtersize1": 3,
        "filtersize": 3,
        "leakyrelu": False,
        "dropoutprob": 0,
    }

    model = load_unet_model(
        str(pth_file),
        n_channels=1,
        config=config,
        device=torch.device("cpu"),
    )
    assert model is not None

    x = torch.zeros(1, 1, 256, 256)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 1, 256, 256)
