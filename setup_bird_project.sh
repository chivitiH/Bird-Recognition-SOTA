#!/bin/bash

# Couleurs pour les messages
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Configuration du projet Bird Recognition ===${NC}"

# 1. Créer la structure de dossiers moderne
echo -e "${GREEN}Création de la structure de dossiers...${NC}"
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/models
mkdir -p logs
mkdir -p checkpoints

# 2. Créer le fichier de configuration moderne
cat > config.py << 'PYEOF'
import os
from pathlib import Path

# Chemins de base
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Dataset Kaggle
KAGGLE_DATASET = "gpiosenka/100-bird-species"
DATASET_RAW_PATH = RAW_DATA_DIR / "dataset_birds_original"
DATASET_CLEAN_PATH = PROCESSED_DATA_DIR / "dataset_birds_clean"

# Paramètres de preprocessing
TARGET_SIZE = (224, 224)
BALANCE_THRESHOLD = 160  # Nombre d'images par classe après équilibrage
TRAIN_SPLIT = 0.70
VALID_SPLIT = 0.15
TEST_SPLIT = 0.15

# Paramètres d'entraînement
BATCH_SIZE = 32
EPOCHS_FREEZE = 15
EPOCHS_FINETUNE = 15
LEARNING_RATE = 1e-4
RANDOM_SEED = 42

# Modèles disponibles
AVAILABLE_MODELS = {
    'efficientnetb0': {'input_size': 224, 'weights': 'imagenet'},
    'efficientnetb7': {'input_size': 224, 'weights': 'imagenet'},
    'mobilenetv2': {'input_size': 224, 'weights': 'imagenet'},
}

def create_dirs():
    """Crée tous les dossiers nécessaires"""
    for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                     MODELS_DIR, LOGS_DIR, CHECKPOINTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    print("✓ Dossiers créés")

if __name__ == "__main__":
    create_dirs()
PYEOF

# 3. Créer le script de téléchargement Kaggle
cat > download_dataset.py << 'PYEOF'
import os
import zipfile
from pathlib import Path
import config

def download_kaggle_dataset():
    """Télécharge le dataset depuis Kaggle"""
    print("📥 Téléchargement du dataset depuis Kaggle...")
    
    # Vérifier que Kaggle API est configurée
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("❌ Kaggle API non configurée!")
        print("Instructions:")
        print("1. Aller sur https://www.kaggle.com/settings")
        print("2. Créer un nouveau API token")
        print("3. Placer kaggle.json dans ~/.kaggle/")
        print("4. chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    config.create_dirs()
    
    # Télécharger avec kaggle CLI
    import subprocess
    
    try:
        cmd = [
            "kaggle", "datasets", "download", 
            "-d", config.KAGGLE_DATASET,
            "-p", str(config.RAW_DATA_DIR),
            "--unzip"
        ]
        subprocess.run(cmd, check=True)
        print("✓ Dataset téléchargé et extrait")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors du téléchargement: {e}")
        return False
    except FileNotFoundError:
        print("❌ Kaggle CLI non installé. Installer avec: pip install kaggle")
        return False

if __name__ == "__main__":
    download_kaggle_dataset()
PYEOF

echo -e "${GREEN}✓ Scripts de configuration créés${NC}"
echo -e "${BLUE}Prochaines étapes:${NC}"
echo "1. Configurer Kaggle API si nécessaire"
echo "2. python download_dataset.py"
echo "3. python preprocessing_pipeline.py"
echo "4. python train.py"

