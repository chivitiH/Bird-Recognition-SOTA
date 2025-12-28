from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

KAGGLE_DATASET = "vinjamuripavan/bird-species"
DATASET_RAW_PATH = RAW_DATA_DIR / "BIRDS 525 SPECIES"
DATASET_CLEAN_PATH = DATASET_RAW_PATH

TARGET_SIZE = (224, 224)
BATCH_SIZE = 384  # MAX VRAM !
EPOCHS_FREEZE = 15
EPOCHS_FINETUNE = 15
LEARNING_RATE = 1e-4
RANDOM_SEED = 42

# Classes à exclure
EXCLUDED_CLASSES = []  # On remplira après avoir trouvé la classe piège

def create_dirs():
    for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                     MODELS_DIR, LOGS_DIR, CHECKPOINTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    print("✓ Dossiers créés")

if __name__ == "__main__":
    create_dirs()
