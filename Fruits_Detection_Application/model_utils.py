"""
model_utils.py
Contains:
  1. SEBlock              — Squeeze-and-Excitation attention module
  2. build_model()        — reconstructs the exact architecture used in training
  3. load_model()         — loads best_model.pth checkpoint
  4. GradCAM class        — forward/backward hooks for Grad-CAM heatmap generation
  5. predict()            — runs inference, returns prediction + confidence + probs
  6. generate_gradcam()   — runs Grad-CAM and returns an overlay image
  
This file mirrors the exact architecture from the training notebook so that
state_dict keys match perfectly when loading best_model.pth.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image


# 1. SE Block — must match training architecture exactly
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (Hu et al., 2018).
    Recalibrates channel-wise feature responses adaptively.
    """
    def __init__(self, channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction_ratio, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction_ratio, 1), channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        z = self.squeeze(x)
        s = self.excitation(z)
        s = s.view(b, c, 1, 1)
        return x * s


# 2. Build model — identical to training notebook
def build_model(num_classes, device):
    """
    Reconstructs MobileNetV3-Small + SE Block + custom classifier head.
    Must match the training notebook exactly for state_dict to load correctly.
    """
    model = models.mobilenet_v3_small(weights=None)
                                                       # we load OUR trained weights next

    # Inject SE Block after the last conv block (576 channels)
    last_block = model.features[-1]
    model.features[-1] = nn.Sequential(
        last_block,
        SEBlock(channels=576, reduction_ratio=16)
    )

    # Replace classifier head
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model.to(device)

# 3. Load checkpoint
def load_model(checkpoint_path, device):
    """
    Loads best_model.pth and returns (model, class_names, idx_to_class).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    class_to_idx = checkpoint["class_to_idx"]
    num_classes  = checkpoint["num_classes"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names  = [idx_to_class[i] for i in range(num_classes)]

    model = build_model(num_classes, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names, idx_to_class


def load_ablation_model(checkpoint_path, device, num_classes):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_model(num_classes, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    used_augmentation = checkpoint.get("augmentation", None)
    return model, used_augmentation


# 4. Preprocessing — identical to base_transform used in training
def get_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    ])

# 5. Standard inference — prediction + confidence + full distribution
def predict(model, image_pil, transform, class_names, device):
    model.eval()
    tensor = transform(image_pil).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]  # [NUM_CLASSES]

    confidence, class_idx = probs.max(dim=0)
    predicted_class = class_names[class_idx.item()]

    return predicted_class, confidence.item(), probs.cpu().numpy()


# 6. Grad-CAM implementation using forward + backward hooks
class GradCAM:
    """
    Implements Gradient-weighted Class Activation Mapping.

    How it works:
      1. Forward hook  : captures the activation maps A at the target layer
      2. Backward hook : captures the gradient of the output w.r.t. A
      3. Weight        : alpha_k = global-average-pool of the gradient (per channel)
      4. Combine       : heatmap = ReLU( sum_k( alpha_k * A_k ) )
      5. Upsample       : resize heatmap to input image size, overlay as a colormap
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Forward hook: store the activation map output of the target layer
        self.fwd_handle = target_layer.register_forward_hook(self._save_activation)

        # Backward hook: store the gradient flowing back into the target layer
        self.bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        # output: feature map tensor [B, C, H, W] from the target conv layer
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output[0]: gradient of the loss w.r.t. this layer's output [B, C, H, W]
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
    
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backward pass: only backprop the score of the target class
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Compute channel-wise weights: global average pool of gradients
        # gradients shape: [1, C, H, W]  →  weights shape: [C]
        weights = self.gradients.mean(dim=(2, 3))[0]   # [C]
        activations = self.activations[0]               # [C, H, W]

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU: keep only features that POSITIVELY influence the class score
        cam = F.relu(cam)

        # Normalize to [0, 1] for visualization
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy(), class_idx

    def remove_hooks(self):
        """Call this when done to avoid memory leaks from dangling hooks."""
        self.fwd_handle.remove()
        self.bwd_handle.remove()


def generate_gradcam_overlay(model, image_pil, transform, target_layer, device,
                              class_idx=None, alpha=0.45):
    # Preprocess
    input_tensor = transform(image_pil).unsqueeze(0).to(device)

    # Run Grad-CAM
    cam_extractor = GradCAM(model, target_layer)
    heatmap, used_class_idx = cam_extractor.generate(input_tensor, class_idx)
    cam_extractor.remove_hooks()

    # Resize heatmap to match original image dimensions
    img_resized = image_pil.resize((224, 224))
    img_np = np.array(img_resized).astype(np.uint8)

    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend original image + heatmap
    overlay_img = (img_np * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)

    return overlay_img, used_class_idx
