import tensorflow as tf
import os, warnings
from pathlib import Path
import config

# SUPPRIMER TOUS LES WARNINGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

tf.keras.mixed_precision.set_global_policy('mixed_float16')

TRAIN_DIR = str(config.DATASET_CLEAN_PATH / "train")
VAL_DIR = str(config.DATASET_CLEAN_PATH / "valid")
TEST_DIR = str(config.DATASET_CLEAN_PATH / "test")

BATCH_SIZE = 512  # Plus gros pour saturer VRAM

print("=" * 60)
print("🐦 BIRD CLASSIFICATION - 524 SPECIES")
print("=" * 60)

from tensorflow.keras.applications.efficientnet import preprocess_input

def build_dataset(directory, batch_size, training=False):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory, image_size=(224, 224), batch_size=batch_size,
        label_mode='int', shuffle=training
    )
    ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=2)
    return ds.prefetch(2)

train_ds = build_dataset(TRAIN_DIR, BATCH_SIZE, training=True)
val_ds = build_dataset(VAL_DIR, BATCH_SIZE)
test_ds = build_dataset(TEST_DIR, BATCH_SIZE)

print(f"\n✓ Batch size: {BATCH_SIZE}")
print(f"✓ Classes: 524\n")

base = tf.keras.applications.EfficientNetB0(
    include_top=False, weights='imagenet', input_shape=(224, 224, 3)
)
base.trainable = False

model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(524, dtype='float32')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

print("⚡ PHASE 1: Frozen base (15 epochs)")
print("-" * 60)

model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=15,
    verbose=1  # Barre de progression !
)

print("\n⚡ PHASE 2: Fine-tuning (15 epochs)")
print("-" * 60)

base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    verbose=1
)

print("\n�� Evaluation...")
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"Test accuracy: {test_acc*100:.2f}%\n")

model.save(config.MODELS_DIR / "bird_524_final.keras")
print("✅ Training complete!\n")
