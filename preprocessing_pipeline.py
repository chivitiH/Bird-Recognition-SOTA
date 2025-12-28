"""
Preprocessing pour Caltech-UCSD Birds-200-2011
Utilise le split train/test existant
"""
import shutil
from pathlib import Path
import config
import random

class CUBPreprocessing:
    def __init__(self, raw_path, output_path):
        self.raw_path = Path(raw_path)
        self.output_path = Path(output_path)
        random.seed(config.RANDOM_SEED)
        
    def run(self):
        print("=" * 60)
        print("🐦 PREPROCESSING CUB-200-2011")
        print("=" * 60)
        
        # Lire le split train/test
        split_file = self.raw_path / "train_test_split.txt"
        train_test_split = {}
        
        with open(split_file, 'r') as f:
            for line in f:
                img_id, is_train = line.strip().split()
                train_test_split[img_id] = int(is_train)
        
        # Lire les images et leurs classes
        images_file = self.raw_path / "images.txt"
        image_paths = {}
        
        with open(images_file, 'r') as f:
            for line in f:
                img_id, img_path = line.strip().split()
                image_paths[img_id] = img_path
        
        # Créer structure de sortie
        train_dir = self.output_path / "train"
        valid_dir = self.output_path / "valid"
        test_dir = self.output_path / "test"
        
        for d in [train_dir, valid_dir, test_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        print("\n📁 Organising images...")
        
        # Séparer train/test selon le fichier
        train_images = {k: v for k, v in image_paths.items() if train_test_split[k] == 1}
        test_images = {k: v for k, v in image_paths.items() if train_test_split[k] == 0}
        
        # Créer validation depuis train (15%)
        train_list = list(train_images.items())
        random.shuffle(train_list)
        
        n_valid = int(len(train_list) * 0.15)
        valid_images = dict(train_list[:n_valid])
        train_images = dict(train_list[n_valid:])
        
        # Copier les images
        self._copy_images(train_images, train_dir)
        self._copy_images(valid_images, valid_dir)
        self._copy_images(test_images, test_dir)
        
        self.display_stats()
        
    def _copy_images(self, image_dict, target_dir):
        """Copie images vers structure train/valid/test"""
        for img_id, img_path in image_dict.items():
            src = self.raw_path / "images" / img_path
            
            # Extraire nom de classe du chemin
            class_name = img_path.split('/')[0]
            class_dir = target_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            # Nom de fichier
            img_name = img_path.split('/')[1]
            dst = class_dir / img_name
            
            shutil.copy2(src, dst)
    
    def display_stats(self):
        print("\n" + "=" * 60)
        print("📊 STATISTIQUES")
        print("=" * 60)
        
        for split in ['train', 'valid', 'test']:
            split_path = self.output_path / split
            if split_path.exists():
                n_classes = len(list(split_path.iterdir()))
                n_images = sum(len(list(c.iterdir())) for c in split_path.iterdir())
                print(f"{split.upper():6s}: {n_classes:3d} classes, {n_images:6d} images")

def main():
    config.create_dirs()
    
    if not config.DATASET_RAW_PATH.exists():
        print("❌ Dataset brut non trouvé!")
        return
    
    pipeline = CUBPreprocessing(
        raw_path=config.DATASET_RAW_PATH,
        output_path=config.DATASET_CLEAN_PATH
    )
    
    pipeline.run()
    print("\n✅ Preprocessing terminé!")

if __name__ == "__main__":
    main()
