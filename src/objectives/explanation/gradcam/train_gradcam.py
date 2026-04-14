import torch
import torch.nn as nn

# ==============================
# GradCAM Training
# ==============================
def replace_relu_with_softplus(module, beta=10):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.Softplus(beta=beta))
        else:
            replace_relu_with_softplus(child, beta)

def replace_softplus_with_relu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Softplus):
            setattr(module, name, nn.ReLU(inplace=True))
        else:
            replace_softplus_with_relu(child)


class TrainableGradCAMPP:
    def __init__(self, model, target_layer):
        self.activations = None
        self.hook = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _, __, output):
        self.activations = output
        if not self.activations.requires_grad:
            self.activations.requires_grad_(True)

    def remove(self):
        self.hook.remove()

    def __call__(self, logits, class_idx, create_graph=True):
        """
        logits: [B, num_classes]
        class_idx: [B]
        create_graph=True  -> training mode (second-order grads)
        create_graph=False -> evaluation mode (no higher-order graph)
        """
        B = logits.size(0)

        scores = logits[torch.arange(B, device=logits.device),class_idx]

        grads = torch.autograd.grad(
            outputs=scores,
            inputs=self.activations,
            grad_outputs=torch.ones_like(scores),
            create_graph=create_graph,
            retain_graph=True
        )[0]

        grads_2 = grads ** 2
        grads_3 = grads_2 * grads

        denom = 2 * grads_2 + torch.abs(self.activations) * grads_3 + 1e-8
        alpha = grads_2 / denom

        weights = (alpha * torch.relu(grads)).sum(dim=(2, 3), keepdim=True)

        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)

        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)

        return cam