# xraymodel.py
from PIL import Image
import numpy as np
import onnxruntime as ort
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Load model
def load_onnx(path="tuberModel.onnx"):
    return ort.InferenceSession(path)

# Preprocess & predict
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_onnx(img, sess):
    x = preprocess(img).unsqueeze(0).numpy()
    inp = sess.get_inputs()[0].name
    out = sess.run(None, {inp: x})[0]
    probs = np.exp(out) / np.exp(out).sum(-1, keepdims=True)
    return probs[0, 1], int(np.argmax(out))

# Occlusion heatmap
def occlusion_heatmap(img: Image.Image, sess, patch_size=30, stride=15):
    base_prob, _ = predict_onnx(img, sess)
    hmap = np.zeros((224, 224), dtype=np.float32)
    count = np.zeros_like(hmap)
    img_np = np.array(img.resize((224, 224)))
    mean_pixel = img_np.mean(axis=(0, 1), keepdims=True).astype(np.uint8)

    for y in range(0, 224, stride):
        for x in range(0, 224, stride):
            occl = img_np.copy()
            y2, x2 = min(224, y + patch_size), min(224, x + patch_size)
            occl[y:y2, x:x2] = mean_pixel
            prob, _ = predict_onnx(Image.fromarray(occl), sess)
            hmap[y:y2, x:x2] += (base_prob - prob)
            count[y:y2, x:x2] += 1

    hmap = hmap / np.maximum(count, 1)
    hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
    return hmap

# Overlay heatmap
def overlay_heatmap(img: Image.Image, heatmap: np.ndarray, alpha=0.5):
    orig = np.array(img.resize((224, 224))) / 255.0
    hm = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)[..., ::-1] / 255.0
    return Image.fromarray(np.uint8(((1 - alpha) * orig + alpha * hm) * 255))
