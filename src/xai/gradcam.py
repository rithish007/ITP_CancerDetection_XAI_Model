
import numpy as np
import cv2
import torch.nn.functional as F


def generate_heatmap(model, image_tensor, target_layer, class_idx=None):
    # Storage for hook outputs.
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

    # Hooks removed
    fwd_handle.remove()
    bwd_handle.remove()

    # Compute Grad-CAM
    acts  = activations[0].detach()
    grads = gradients[0].detach()

    # Global average pool the gradients over the spatial dims
    weights = grads.mean(dim=(2, 3), keepdim=True)

    # Weighted sum of feature maps
    cam = (weights * acts).sum(dim=1, keepdim=True)

    # ReLU
    cam = F.relu(cam)

    # Upsample the small spatial map to the input image size
    h, w = image_tensor.shape[2], image_tensor.shape[3]
    cam  = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)

    # Normalise to [0, 1]
    cam = cam.squeeze().cpu().numpy()   # [H, W]
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam.astype(np.float32)


def overlay_heatmap(heatmap, original_image, alpha=0.45):

    # Scale and apply heatmap
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    coloured_bgr  = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    coloured_rgb  = cv2.cvtColor(coloured_bgr, cv2.COLOR_BGR2RGB)

    if original_image.dtype != np.uint8:
        original_image = (original_image * 255).clip(0, 255).astype(np.uint8)

    # Alpha controls the heatmap transparency
    overlay = cv2.addWeighted(coloured_rgb, alpha, original_image, 1 - alpha, 0)
    return overlay
