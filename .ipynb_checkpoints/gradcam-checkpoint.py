import torch
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.grads = None
        self.activations = None

        target_layer.register_forward_hook(
            # capture the layer's outputs on the forward pass
            lambda m, i, o: setattr(self, 'activations', o.detach())
        )
        target_layer.register_full_backward_hook(
            # capture the layer's gradients on the backward pass
            lambda m, gi, go: setattr(self, 'grads', go[0].detach())
        )

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        predicted = output.argmax(dim=1).item()
        if class_idx is None:
            class_idx = predicted

        # backpropagation for a specific class
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot)

        # pools the gradients across height and width
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        # scale each feature map channel according to importance
        # collapse channels into a single map
        cam = torch.relu((weights * self.activations).sum(dim=1)).squeeze().cpu().numpy()

        # normalise to [0, 1] so can be visualised on heatmap
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam, predicted

class GradCamPlusPlus(GradCAM):
    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        predicted = output.argmax(dim=1).item()
        if class_idx is None:
            class_idx = predicted

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot)

        # uses second and third order gradients to calculate weights
        grads    = self.grads
        acts     = self.activations
        grads_sq = grads ** 2
        alpha    = grads_sq / (2 * grads_sq + grads ** 3 * acts.sum(dim=(2,3), keepdim=True) + 1e-7)
        weights  = (alpha * torch.relu(output[0, class_idx].exp() * grads)).sum(dim=(2,3), keepdim=True)

        cam = torch.relu((weights * acts).sum(dim=1)).squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam, predicted

# function: overlay_cam
def overlay_cam(image_np, cam, alpha=0.5):
    cam_resized = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return np.uint8(alpha * heatmap + (1 - alpha) * image_np)