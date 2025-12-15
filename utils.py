import os
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from collections import Counter

def compute_metrics(y_true, y_pred, average='macro'):
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    return {'f1': f1, 'precision': prec, 'recall': rec}

def get_class_weights_from_labels(labels, num_classes=None):
    counts = Counter(labels)
    if num_classes is None:
        num_classes = max(counts.keys()) + 1
    freq = np.array([counts.get(i, 0) for i in range(num_classes)], dtype=np.float32)
    # avoid div by zero
    freq = np.where(freq == 0, 1.0, freq)
    weights = 1.0 / freq
    weights = weights / weights.sum() * num_classes
    return weights

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(state, best_path)

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean', ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction='none', ignore_index=ignore_index)

    def forward(self, inputs, targets):
        logpt = -self.ce(inputs, targets)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(6,6), cmap='Blues'):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    if class_names:
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
    plt.tight_layout()
    return fig

# Minimal Grad-CAM utility reuse (safe remove hooks)
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        # register_full_backward_hook for newer PyTorch versions? fallback
        try:
            self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))
        except Exception:
            self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def __call__(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        loss = output[0, class_idx]
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        grads = self.gradients[0]  # C x H x W
        acts = self.activations[0]  # C x H x W
        weights = grads.mean(dim=(1, 2), keepdim=True)  # C x1x1
        cam = (weights * acts).sum(dim=0).cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()
        return cam

    def remove_hooks(self):
        for h in self.hook_handles:
            try:
                h.remove()
            except Exception:
                pass

def overlay_cam_on_image(img_rgb, cam, colormap=cv2.COLORMAP_JET, alpha=0.5):
    h, w, _ = img_rgb.shape
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (w, h))
    overlay = np.uint8(img_rgb * 255 * (1 - alpha) + heatmap * alpha)
    return overlay