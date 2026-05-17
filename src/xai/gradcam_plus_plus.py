
import numpy as np
import torch.nn.functional as F
from src.xai.gradcam import overlay_heatmap   # re-used directly, no duplication


def generate_heatmap(model, image_tensor, target_layer, class_idx=None):

    activations = []
    gradients   = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    logits = model(image_tensor)

    if class_idx is None:
        class_idx = logits.argmax(dim=1).item()

    # Backward pass
    model.zero_grad()
    logits[0, class_idx].backward()

    fwd_handle.remove()
    bwd_handle.remove()

    acts  = activations[0].detach()   # [1, C, H', W']
    grads = gradients[0].detach()     # [1, C, H', W']

    # Grad-CAM++ weight computation
    grads_sq = grads.pow(2)
    grads_cu = grads.pow(3)

    alpha_num = grads_sq
    alpha_den = (
        2.0 * grads_sq
        + (acts * grads_cu).sum(dim=(2, 3), keepdim=True).clamp(min=1e-7)
    )

    alpha   = alpha_num / alpha_den
    weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

    # Weighted activation map, upsample, normalise
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = F.relu(cam)

    h, w = image_tensor.shape[2], image_tensor.shape[3]
    cam  = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)

    cam = cam.squeeze().cpu().numpy()
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam.astype(np.float32)