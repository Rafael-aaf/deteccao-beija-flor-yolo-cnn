import matplotlib
import numpy as np
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._on_forward)

    def _on_forward(self, module, inputs, output):
        self.activations = output.detach()
        output.register_hook(self._on_backward)

    def _on_backward(self, grad):
        self.gradients = grad.detach()

    def __call__(self, image, class_index=None):
        self.model.zero_grad()
        logits = self.model(image)
        probs = logits.softmax(dim=1)[0].detach().cpu().numpy()
        if class_index is None:
            class_index = int(probs.argmax())
        logits[0, class_index].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        return (cam / cam.max().clamp_min(1e-8)).cpu().numpy(), probs


def overlay(image: np.ndarray, cam: np.ndarray, alpha: float, colormap: str) -> np.ndarray:
    heat = matplotlib.colormaps[colormap](cam)[..., :3]
    return (1 - alpha) * image + alpha * heat