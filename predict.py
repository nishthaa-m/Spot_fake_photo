"""Usage:
    python predict.py some_image.jpg
Prints ONE number from 0 to 1:
    0 = real photo,  1 = photo of a screen (recapture / fraud)
"""

import sys
import os
import pickle
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Find model assets relative to this script
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ASSETS_PATH = os.path.join(SCRIPT_DIR, "model_assets.pkl")

# Pre-load model and preprocess transform once at module level to make repeated calls fast (if imported)
_model = None
_scaler = None
_classifier = None
_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_assets():
    global _model, _scaler, _classifier
    if _model is None:
        # Load pre-trained MobileNetV3-Large feature extractor
        # It loads from local PyTorch cache (~0.3s)
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        _model = torch.nn.Sequential(model.features, model.avgpool)
        _model.eval()
        
    if _scaler is None or _classifier is None:
        if not os.path.exists(ASSETS_PATH):
            raise FileNotFoundError(f"Model assets file not found at {ASSETS_PATH}. Run train.py first.")
        with open(ASSETS_PATH, "rb") as f:
            assets = pickle.load(f)
            _scaler = assets["scaler"]
            _classifier = assets["classifier"]

def predict(image_path: str) -> float:
    """Predicts whether the image is a real photo (0) or a photo of a screen (1)."""
    # Ensure assets are loaded
    load_assets()
    
    # Load and crop image at native resolution
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    crop_sz = 224
    
    # Extract 5 crops (center, top-left, top-right, bottom-left, bottom-right)
    if w < crop_sz or h < crop_sz:
        img_resized = img.resize((crop_sz, crop_sz))
        crops = [img_resized] * 5
    else:
        crops = [
            img.crop((w//2 - crop_sz//2, h//2 - crop_sz//2, w//2 + crop_sz//2, h//2 + crop_sz//2)),
            img.crop((0, 0, crop_sz, crop_sz)),
            img.crop((w - crop_sz, 0, w, crop_sz)),
            img.crop((0, h - crop_sz, crop_sz, h)),
            img.crop((w - crop_sz, h - crop_sz, w, h))
        ]
        
    # Batch preprocess
    batch = torch.stack([_preprocess(crop) for crop in crops])
    
    # Feature extraction in a single batch forward pass
    with torch.no_grad():
        feats = _model(batch)
        feats = torch.squeeze(feats).numpy()
        # Ensure feats is a 2D array of shape (5, 960)
        if len(feats.shape) == 1:
            feats = feats.reshape(1, -1)
        elif len(feats.shape) > 2:
            feats = feats.reshape(feats.shape[0], -1)
            
    # Scale features
    feats_scaled = _scaler.transform(feats)
    
    # Predict crop probabilities (probability of class 1: screen recapture)
    probs = _classifier.predict_proba(feats_scaled)[:, 1]
    
    # Aggregate probabilities (Mean aggregation)
    fraud_score = float(np.mean(probs))
    return fraud_score

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
        
    try:
        score = predict(sys.argv[1])
        # Print only the final float score to stdout
        print(f"{score:.4f}")
    except Exception as e:
        # Fallback or error print
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)
