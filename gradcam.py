import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.cm as cm
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None

        self._fwd = target_layer.register_forward_hook(self._save_activations)

    def _save_activations(self, module, input, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._save_gradients)
        else:
            # Force requires_grad so hook fires
            output.retain_grad()
            output.register_hook(self._save_gradients)

    def _save_gradients(self, grad):
        self.gradients = grad

    def remove_hooks(self):
        self._fwd.remove()

    def __call__(self, x, class_idx=None):
        self.model.eval()

        with torch.enable_grad():
            x = x.requires_grad_(True)
            output = self.model(x)

            if class_idx is None:
                class_idx = output.argmax(dim=1).item()

            self.model.zero_grad()
            score = output[0, class_idx]
            score.backward()

        if self.gradients is None:
            # Fallback: use activation mean as heatmap (no gradient needed)
            acts = self.activations[0].detach()
            heatmap = acts.mean(dim=0).cpu().numpy()
        else:
            pooled = self.gradients.mean(dim=[0, 2, 3])
            acts   = self.activations[0].clone().detach()
            for i in range(acts.shape[0]):
                acts[i] *= pooled[i]
            heatmap = acts.mean(dim=0).cpu().numpy()

        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap, output.detach()


def generate_gradcam(model, img_tensor, target_class=None):
   
    target_layer = model.get_last_conv_layer()
    gc           = GradCAM(model, target_layer)

    heatmap, prediction = gc(img_tensor, target_class)
    gc.remove_hooks()

    h, w = img_tensor.shape[2], img_tensor.shape[3]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = (cm.jet(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std  = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    orig = img_tensor.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
    orig = np.clip(std * orig + mean, 0, 1)
    orig = (orig * 255).astype(np.uint8)

    overlay = cv2.addWeighted(orig, 0.6, heatmap_colored, 0.4, 0)
    return heatmap, Image.fromarray(overlay), prediction
