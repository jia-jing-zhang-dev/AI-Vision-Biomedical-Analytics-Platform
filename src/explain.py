"""
explain.py
==========
Grad-CAM (Selvaraju et al., ICCV 2017) for the BloodMNIST classifiers:
which parts of the image pushed the model towards its prediction.

The forward pass is split by hand at the last convolutional stage and the
gradient taken with torch.autograd.grad, instead of registering hooks: the
Streamlit app shares one cached model between all visitors, and hooks would
be global state that concurrent requests could trip over.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _features_and_head(model, x):
    """Returns (activations of the last conv stage, function mapping them to logits)."""
    arch = getattr(model, "arch", "cnn")
    if arch in ("resnet18", "resnet18_224"):
        m = model
        a = m.layer4(m.layer3(m.layer2(m.layer1(m.maxpool(m.relu(m.bn1(m.conv1(x))))))))
        return a, lambda acts: m.fc(torch.flatten(m.avgpool(acts), 1))
    if arch == "cnn":
        a = model.features[:-1](x)                         # last conv block, before global pooling
        return a, lambda acts: model.classifier(model.features[-1](acts))
    if arch == "dinobloom_s":
        # ViT: the class token only sees the patch tokens through attention.
        # Activations are the normalized patch tokens entering the last
        # block's attention (blocks[-1].norm1, the layer pytorch-grad-cam uses
        # for ViTs), shaped back into a grid; the residual input is held fixed,
        # so the gradient flows through attention only, as with a hook there.
        m, last = model, model.blocks[-1]
        tokens = m.blocks[:-1](m.norm_pre(m._pos_embed(m.patch_embed(x)))).detach()
        normed = last.norm1(tokens)
        n_pre = m.num_prefix_tokens
        side = int((tokens.shape[1] - n_pre) ** 0.5)
        prefix = normed[:, :n_pre].detach()

        def head(grid):
            patches = grid.reshape(grid.shape[0], grid.shape[1], -1).transpose(1, 2)
            h = tokens + last.drop_path1(last.ls1(last.attn(torch.cat([prefix, patches], dim=1))))
            h = h + last.drop_path2(last.ls2(last.mlp(last.norm2(h))))
            return m.forward_head(m.norm(h))

        grid = normed[:, n_pre:].transpose(1, 2).reshape(x.shape[0], -1, side, side)
        return grid, head
    raise ValueError(f"Grad-CAM is not implemented for architecture {arch!r}")


def grad_cam(model, image, target: int | None = None):
    """Grad-CAM heatmap for one PIL image (any size).

    Returns (heatmap, target): ``heatmap`` is a float32 array in [0, 1] at the
    model's input resolution (28x28 for the CNN, 224x224 for ResNet-18);
    ``target`` is the explained class (default: the predicted one).
    """
    from src.image_io import to_8bit
    from src.medical_cnn import input_transform

    device = next(model.parameters()).device
    x = input_transform(getattr(model, "arch", "cnn"), from_upload=True)(to_8bit(image, "RGB"))
    x = x.unsqueeze(0).to(device)
    with torch.enable_grad():
        acts, head = _features_and_head(model, x)
        acts = acts.detach().requires_grad_(True)
        logits = head(acts)
        if target is None:
            target = int(logits.argmax(1))
        grads, = torch.autograd.grad(logits[0, target], acts)
    weights = grads.mean(dim=(2, 3), keepdim=True)          # importance of each channel
    cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    peak = cam.max()
    return (cam / peak if peak > 0 else cam).astype(np.float32), target


def overlay(image, heatmap: np.ndarray, size: int = 224, alpha: float = 0.45) -> np.ndarray:
    """uint8 RGB picture of ``image`` with the heatmap on top (jet colours)."""
    from matplotlib import colormaps
    from PIL import Image

    from src.image_io import to_8bit

    base = np.asarray(to_8bit(image, "RGB").resize((size, size), Image.Resampling.NEAREST), np.float32)
    heat = np.asarray(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((size, size),
                                                                              Image.Resampling.BILINEAR))
    colored = colormaps["jet"](heat / 255.0)[..., :3] * 255
    return ((1 - alpha) * base + alpha * colored).clip(0, 255).astype(np.uint8)
