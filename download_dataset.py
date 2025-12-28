import subprocess
from pathlib import Path
import config

def download_kaggle_dataset():
    print("📥 Téléchargement du dataset depuis Kaggle...")
    
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("❌ Kaggle API non configurée!")
        return False
    
    config.create_dirs()
    
    try:
        cmd = [
            "kaggle", "datasets", "download", 
            "-d", config.KAGGLE_DATASET,
            "-p", str(config.RAW_DATA_DIR),
            "--unzip"
        ]
        subprocess.run(cmd, check=True)
        print("✓ Dataset téléchargé!")
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    download_kaggle_dataset()
