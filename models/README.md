# Denoiser model zoo

This directory is intentionally shipped without trained third-party weights.
Put your `.pt` / `.pth` files here or select them from the GUI.

Supported GUI denoiser families:

- DnCNN
- FFDNet
- DRUNet
- SCUNet
- Model zoo / custom manifest

A weights file may be either:

```python
torch.save(model.state_dict(), "dncnn_gray.pth")
```

or a checkpoint:

```python
torch.save({
    "state_dict": model.state_dict(),
    "epoch": 120,
    "loss": 0.0014,
}, "dncnn_gray_checkpoint.pth")
```

For automatic architecture selection, place a JSON file next to the weights with
the same stem, for example:

```text
drunet_gray.pth
drunet_gray.json
```

Example manifest:

```json
{
  "architecture": "DRUNet",
  "channels": 1,
  "features": 64,
  "depth": 4,
  "residual": true,
  "clamp": true
}
```

`residual: true` means the network predicts noise/residual and the denoised image is `x - model(x)`.
`residual: false` means the network output is treated directly as the denoised image.
