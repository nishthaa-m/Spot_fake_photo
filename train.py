import os
import time
import pickle
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def get_mobilenet_v3():
    # Load pre-trained MobileNetV3-Large
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    # Freeze all layers and extract feature representation
    model = torch.nn.Sequential(model.features, model.avgpool)
    model.eval()
    return model

def extract_5_crops(img_path, model, preprocess):
    """Extracts 5 native-resolution crops and their MobileNetV3 features."""
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    crop_sz = 224
    
    # Define crops: center, top-left, top-right, bottom-left, bottom-right
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
        
    crop_feats = []
    for crop in crops:
        tensor = preprocess(crop).unsqueeze(0)
        with torch.no_grad():
            feat = model(tensor)
            feat = torch.squeeze(feat).numpy().flatten()
            crop_feats.append(feat)
            
    return np.array(crop_feats) # Shape: (5, 960)

def main():
    data_dir = r"C:\Users\nisht\OneDrive\Desktop\data"
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    print("Loading pre-trained MobileNetV3...")
    model = get_mobilenet_v3()
    
    classes = ["real", "screen"]
    X_list = []
    y_list = []
    
    t0 = time.time()
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.exists(cls_dir):
            print(f"Directory not found: {cls_dir}")
            continue
            
        print(f"Loading and processing images from {cls_dir}...")
        files = os.listdir(cls_dir)
        for idx, file_name in enumerate(files):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')) and file_name.lower() != "desktop.ini":
                path = os.path.join(cls_dir, file_name)
                try:
                    # Extract 5 crops' features
                    feats = extract_5_crops(path, model, preprocess)
                    if feats is not None and feats.shape == (5, 960):
                        X_list.append(feats)
                        y_list.extend([0 if cls == "real" else 1] * 5)
                except Exception as e:
                    print(f"Error loading {file_name}: {e}")
                    
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(files)} images")
                
    X = np.vstack(X_list) # Shape: (N * 5, 960)
    y = np.array(y_list)   # Shape: (N * 5,)
    
    print(f"\nFeature extraction complete. Feature matrix shape: {X.shape}")
    print(f"Time taken: {time.time() - t0:.2f} seconds")
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train L2-regularized Logistic Regression (C=0.01)
    print("Training Logistic Regression classifier...")
    clf = LogisticRegression(max_iter=1000, C=0.01, random_state=42)
    clf.fit(X_scaled, y)
    
    # Save scaler and classifier assets
    assets_path = "model_assets.pkl"
    print(f"Saving model assets to {assets_path}...")
    with open(assets_path, "wb") as f:
        pickle.dump({
            "scaler": scaler,
            "classifier": clf
        }, f)
        
    print("Training pipeline finished successfully!")

if __name__ == "__main__":
    main()
