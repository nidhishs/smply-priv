import numpy as np
import torch
from torch.cuda.amp import autocast
from torchvision import transforms
from tqdm import tqdm

from e2fgvi.model import InpaintGenerator

# Stacks a list of np.ndarray images into a single tensor.
_TRANSFORM = transforms.Compose(
    [
        lambda x: np.stack(
            x if x[0].ndim == 3 else [np.expand_dims(i, -1) for i in x], axis=0
        ),
        lambda x: torch.stack([transforms.ToTensor()(i) for i in x], dim=0),
    ]
)
_STEPS = 10
_NEIGHBOR_STRIDE = 5


def setup_e2fgvi(ckpt_path, device="cuda"):
    model = InpaintGenerator().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    return model


@torch.no_grad()
def run_e2fgvi(
    model, 
    frames,
    masks,
    neighbor_stride=_NEIGHBOR_STRIDE,
    steps=_STEPS,
    device='cuda',
    progress=False,
):
    """
    Runs the E2FGVI in-painting model on the given frames and masks.

    Args:
        model (nn.Module or str): The E2FGVI model to use. If str, it should be the path to a checkpoint.
        frames (List[np.ndarray]): A list of frames to in-paint.
        masks (List[np.ndarray]): A list of masks corresponding to the frames.
        neighbor_stride (int): The number of frames to skip between neighbors.
        steps (int): The number of frames to skip between reference frames.
        device (str): The device to run the model on.
        progress (bool): Whether to display a progress bar.

    Returns:
        List[np.ndarray]: The in-painted frames.
    """
    if isinstance(model, str):
        model = setup_e2fgvi(model, device)

    (height, width), num_frames = frames[0].shape[:2], len(frames)
    binary_masks = [np.expand_dims(m != 0, 2).astype(np.uint8) for m in masks]
    composite_frames = [None] * num_frames

    _it = range(0, num_frames, neighbor_stride)
    for f in tqdm(_it) if progress else _it:
        neighbor_idx = list(
            range(max(0, f - neighbor_stride), min(num_frames, f + neighbor_stride + 1))
        )
        reference_idx = [
            i for i in range(0, num_frames, steps) if i not in neighbor_idx
        ]

        selected_masks = (
            _TRANSFORM([masks[i] for i in neighbor_idx + reference_idx])
            .unsqueeze(0)
            .to(device)
        )
        selected_frames = (
            _TRANSFORM([frames[i] for i in neighbor_idx + reference_idx])
            .unsqueeze(0)
            .to(device)
        ) * 2 - 1  # Normalize to [-1, 1]

        masked_frames = selected_frames * (1 - selected_masks)
        padded_frames = _get_padded_frames(masked_frames, height, width)
        predicted_frames = _get_predicted_frames(
            model, padded_frames, len(neighbor_idx), height, width
        )
        # Convert from (B, C, H, W) to (B, H, W, C)
        predicted_frames = predicted_frames.cpu().permute(0, 2, 3, 1).numpy() * 255
        composite_frames = _update_composite_frames(
            composite_frames, predicted_frames, frames, binary_masks, neighbor_idx
        )

    return composite_frames


def _get_padded_frames(frames, h, w, mod_size_h=60, mod_size_w=108):
    h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
    w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
    padded_frames = torch.cat([frames, torch.flip(frames, [3])], 3)[
        :, :, :, : h + h_pad, :
    ]
    padded_frames = torch.cat([padded_frames, torch.flip(padded_frames, [4])], 4)[
        :, :, :, :, : w + w_pad
    ]
    return padded_frames


@autocast()
def _get_predicted_frames(model, frames, num_neighbors, h, w):
    predicted_frames, _ = model(frames, num_neighbors)
    predicted_frames = predicted_frames[:, :, :h, :w]
    predicted_frames = (predicted_frames + 1) / 2
    return predicted_frames


def _update_composite_frames(
    composite_frames, predicted_frames, original_frames, binary_masks, neighbor_idx
):
    for i in range(len(neighbor_idx)):
        idx = neighbor_idx[i]
        img = predicted_frames[i] * binary_masks[idx] + original_frames[idx] * (
            1 - binary_masks[idx]
        )
        if composite_frames[idx] is None:
            composite_frames[idx] = img
        else:
            composite_frames[idx] = (
                composite_frames[idx].astype(np.float32) * 0.5
                + img.astype(np.float32) * 0.5
            )
    return composite_frames
