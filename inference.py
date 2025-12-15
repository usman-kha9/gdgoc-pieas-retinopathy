import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from dataset import get_transforms
from model import CustomCNN
from utils import GradCAM, overlay_cam_on_image

def load_model(path, device, num_classes=5):
    checkpoint = torch.load(path, map_location=device)
    model = CustomCNN(num_classes=num_classes)
    state = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

def preprocess_image(img_path, image_size):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tf = get_transforms(image_size, False)
    augmented = tf(image=img)
    tensor = augmented['image'].unsqueeze(0)
    return tensor, img

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_path, device, num_classes=args.num_classes)
    tensor, orig_img = preprocess_image(args.img_path, args.image_size)
    tensor = tensor.to(device).float()
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred = probs.argmax(dim=1).item()
        confidence = probs.max().item()
    print(f"Predicted class: {pred}  confidence: {confidence:.4f}")

    if args.use_gradcam:
        # hook to last conv in model: use block4.conv's last conv (approx)
        target_layer = model.block4.conv[-3] if hasattr(model.block4, 'conv') else None
        if target_layer is None:
            print("Unable to find target layer for Grad-CAM.")
            return
        cam = GradCAM(model, target_layer)
        cam_map = cam(tensor, class_idx=pred)
        overlay = overlay_cam_on_image(orig_img / 255.0, cam_map, alpha=0.5)
        out_path = args.output_path or (args.img_path + '.gradcam.jpg')
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cam.remove_hooks()
        print("Saved Grad-CAM overlay to:", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--use_gradcam', type=bool, default=True)
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()
    main(args)