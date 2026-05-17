
import numpy as np
import torch
import torch.nn.functional as F
from src.xai.gradcam import overlay_heatmap   # re-used directly, no duplication


def generate_heatmap(model, image_tensor, target_layer, class_idx=None):

    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    handle = target_layer.register_forward_hook(forward_hook)

    with torch.no_grad():
        model(image_tensor)

    handle.remove()

    # Feature maps
    acts = activations[0].detach().squeeze(0)
    C, Hf, Wf = acts.shape

    # Flatten spatial dims and mean-centre across the channel axis
    acts_flat = acts.view(C, -1)
    acts_flat = acts_flat - acts_flat.mean(dim=1, keepdim=True)

    # SVD
    try:
        _, _, Vt = torch.linalg.svd(acts_flat, full_matrices=False)
        v1 = Vt[0]

        # Sign correction as SVD is arbitrary
        mean_act = acts_flat.mean(dim=0)
        if (v1 * mean_act).sum() < 0:
            v1 = -v1

        cam = v1.view(Hf, Wf).cpu().numpy()
    except Exception:
        # Fallback to mean activation map if SVD fails numerically
        cam = acts_flat.mean(dim=0).view(Hf, Wf).cpu().numpy()

    # ReLU
    cam = np.maximum(cam, 0)

    # Upsample
    h, w = image_tensor.shape[2], image_tensor.shape[3]
    cam_t = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
    cam_t = F.interpolate(cam_t, size=(h, w), mode="bilinear", align_corners=False)
    cam   = cam_t.squeeze().numpy()

    # Normalise to [0, 1]
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam.astype(np.float32)
