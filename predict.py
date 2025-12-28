import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image
import sys

MODEL_PATH = "data/models/bird_524_98.24percent.keras"
CLASSES_DIR = Path("data/raw/BIRDS 525 SPECIES/train")

# Charger le modèle
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Lire les noms de classes
class_names = sorted([d.name for d in CLASSES_DIR.iterdir() if d.is_dir()])
print(f"✓ Model loaded: {len(class_names)} classes\n")

def predict_image(image_path):
    # Charger et préparer l'image
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, 0)
    
    # Preprocessing EfficientNet
    from tensorflow.keras.applications.efficientnet import preprocess_input
    img_array = preprocess_input(img_array)
    
    # Prédiction
    predictions = model.predict(img_array, verbose=0)
    predictions = tf.nn.softmax(predictions[0])
    
    # Top 5
    top5_idx = np.argsort(predictions)[-5:][::-1]
    
    print(f"📸 Image: {image_path}\n")
    print("Top 5 predictions:")
    print("-" * 50)
    for i, idx in enumerate(top5_idx, 1):
        print(f"{i}. {class_names[idx]:40s} {predictions[idx]*100:5.2f}%")
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    
    predict_image(sys.argv[1])
