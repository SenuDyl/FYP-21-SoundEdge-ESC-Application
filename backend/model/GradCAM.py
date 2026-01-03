import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx=None):
        with torch.enable_grad():
            logits = self.model(x)

        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[:, class_idx].sum()
        self.model.zero_grad(set_to_none=True)
        score.backward()

        # activations: [B, C, H, W], gradients: [B, C, H, W]
        w = self.gradients.mean(dim=(2,3), keepdim=True)   # [B, C, 1, 1]
        cam = (w * self.activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = F.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam, logits
