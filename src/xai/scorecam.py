import numpy as np
import torch
import torch.nn.functional as F


def generate_heatmap(model, image_tensor, target_layer, class_idx=None, batch_size=32):
    activations = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    handle = target_layer.register_forward_hook(forward_hook)

    # One forward pass to get activations + predicted class
    with torch.no_grad():
        logits = model(image_tensor)

    handle.remove()

    if class_idx is None:
        class_idx = logits.argmax(dim=1).item()

    acts = activations[0]
    C    = acts.shape[1]
    h, w = image_tensor.shape[2], image_tensor.shape[3]

    # Baseline forward pass on a zeroed image
    baseline = torch.zeros_like(image_tensor)
    with torch.no_grad():
        baseline_logits = model(baseline)
        baseline_score  = torch.softmax(baseline_logits, dim=1)[0, class_idx].item()

    # Build one soft mask per activation channel
    masks = acts.squeeze(0)
    masks = F.interpolate(
        masks.unsqueeze(1),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    # Per-channel min-max normalisation
    m_min = masks.flatten(1).min(dim=1).values[:, None, None]
    m_max = masks.flatten(1).max(dim=1).values[:, None, None]
    masks = (masks - m_min) / (m_max - m_min + 1e-7)   # [C, H, W]

    # Score each mask in batches
    scores = []
    for start in range(0, C, batch_size):
        end        = min(start + batch_size, C)
        batch_masks = masks[start:end]
        masked_inputs = image_tensor * batch_masks.unsqueeze(1)
        with torch.no_grad():
            out    = model(masked_inputs)
            probs  = torch.softmax(out, dim=1)
        scores.append(probs[:, class_idx].cpu())

    scores = torch.cat(scores)

    weights = (scores - baseline_score).numpy()

    # Weighted sum of activation maps, upsample, normalise
    acts_np  = acts.squeeze(0).cpu().numpy()
    weights  = weights[:, None, None]
    cam      = (weights * acts_np).sum(axis=0)

    cam = np.maximum(cam, 0)

    # Upsample
    cam_t = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)   # [1,1,H',W']
    cam_t = F.interpolate(cam_t, size=(h, w), mode="bilinear", align_corners=False)
    cam   = cam_t.squeeze().numpy()                            # [H, W]

    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam.astype(np.float32)