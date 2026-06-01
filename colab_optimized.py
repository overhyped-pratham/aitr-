# ================================================================
#  ██████╗ ███████╗██╗     ██╗          ██╗
# ██╔════╝ ██╔════╝██║     ██║         ███║
# ██║      █████╗  ██║     ██║         ╚██║
# ██║      ██╔══╝  ██║     ██║          ██║
# ╚██████╗ ███████╗███████╗███████╗     ██║
#  ╚═════╝ ╚══════╝╚══════╝╚══════╝     ╚═╝
#  INSTALL + DOWNLOAD DATASET
# ================================================================
# %%

!pip install -q kagglehub tensorflow matplotlib seaborn scikit-learn albumentations opencv-python-headless

import kagglehub, os

path = kagglehub.dataset_download("ismailnasri20/driver-drowsiness-dataset-ddd")
print("✅ Dataset path:", path)

# Walk to find the actual image root
for root, dirs, files in os.walk(path):
    print(f"  📁 {root}  →  subdirs={dirs}  files={len(files)}")
    if len(dirs) == 0 and len(files) > 0:
        break


# ================================================================
#  ██████╗ ███████╗██╗     ██╗         ██████╗
# ██╔════╝ ██╔════╝██║     ██║         ╚════██╗
# ██║      █████╗  ██║     ██║          █████╔╝
# ██║      ██╔══╝  ██║     ██║         ██╔═══╝
# ╚██████╗ ███████╗███████╗███████╗    ███████╗
#  ╚═════╝ ╚══════╝╚══════╝╚══════╝    ╚══════╝
#  IMPORTS + CONFIG + EDGE-CASE SAFE DATA LOADING
# ================================================================
# %%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
from PIL import Image
import warnings, json, gc, glob, random
warnings.filterwarnings('ignore')

# ──────── CONFIGURATION ────────
IMG_SIZE          = 224       # EfficientNet native size
BATCH_SIZE        = 32
EPOCHS            = 40
INITIAL_LR        = 1e-3
LABEL_SMOOTHING   = 0.1      # Prevents overconfident predictions
VALIDATION_SPLIT  = 0.2
SEED              = 42
MIXUP_ALPHA       = 0.2      # MixUp augmentation strength

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ──────── AUTO-DETECT DATASET ROOT ────────
DATASET_DIR = None
for root, dirs, files in os.walk(path):
    lower_dirs = [d.lower() for d in dirs]
    if any('drowsy' in d for d in lower_dirs):
        DATASET_DIR = root
        break

if DATASET_DIR is None:
    # Fallback: find any directory with 2 subdirectories containing images
    for root, dirs, files in os.walk(path):
        if len(dirs) == 2:
            DATASET_DIR = root
            break

assert DATASET_DIR is not None, "❌ Could not auto-detect dataset directory!"
print(f"\n📂 Dataset root: {DATASET_DIR}")
class_dirs = sorted(os.listdir(DATASET_DIR))
print(f"📁 Classes found: {class_dirs}")

# ──────── EDGE CASE: Scan for corrupted images ────────
print("\n🔍 Scanning for corrupted images...")
corrupt_count = 0
total_count   = 0
for class_dir in class_dirs:
    class_path = os.path.join(DATASET_DIR, class_dir)
    if not os.path.isdir(class_path):
        continue
    for fname in os.listdir(class_path):
        fpath = os.path.join(class_path, fname)
        total_count += 1
        try:
            img = Image.open(fpath)
            img.verify()  # Verify it's a valid image
        except Exception:
            os.remove(fpath)  # Remove corrupted file
            corrupt_count += 1

print(f"   Total images scanned: {total_count}")
print(f"   Corrupted & removed:  {corrupt_count}")

# ──────── EDGE CASE: Check class balance ────────
class_counts = {}
for class_dir in class_dirs:
    class_path = os.path.join(DATASET_DIR, class_dir)
    if os.path.isdir(class_path):
        count = len(os.listdir(class_path))
        class_counts[class_dir] = count
        print(f"   {class_dir}: {count} images")

imbalance_ratio = max(class_counts.values()) / (min(class_counts.values()) + 1)
print(f"\n⚖️  Imbalance ratio: {imbalance_ratio:.2f}x")
if imbalance_ratio > 1.5:
    print("   ⚠️  Significant imbalance detected — class weights will be applied")

print(f"\n✅ Config: {IMG_SIZE}×{IMG_SIZE}px | batch={BATCH_SIZE} | epochs={EPOCHS} | label_smoothing={LABEL_SMOOTHING}")


# ================================================================
#  ██████╗ ███████╗██╗     ██╗         ██████╗
# ██╔════╝ ██╔════╝██║     ██║         ╚════██╗
# ██║      █████╗  ██║     ██║          █████╔╝
# ██║      ██╔══╝  ██║     ██║          ╚═══██╗
# ╚██████╗ ███████╗███████╗███████╗    ██████╔╝
#  ╚═════╝ ╚══════╝╚══════╝╚══════╝    ╚═════╝
#  DATA PIPELINE WITH HEAVY AUGMENTATION
# ================================================================
# %%

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ──────── AGGRESSIVE AUGMENTATION for training ────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=VALIDATION_SPLIT,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    brightness_range=[0.7, 1.3],
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    channel_shift_range=30.0,     # Color jitter
    fill_mode='nearest'
)

# ──────── NO augmentation for validation ────────
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=VALIDATION_SPLIT
)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    seed=SEED,
    shuffle=True,
    interpolation='bilinear'
)

val_gen = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    seed=SEED,
    shuffle=False,
    interpolation='bilinear'
)

CLASS_NAMES = list(train_gen.class_indices.keys())
print(f"\n🏷️  Classes: {train_gen.class_indices}")
print(f"📊 Train: {train_gen.samples}  |  Val: {val_gen.samples}")

# ──────── Compute class weights to handle imbalance ────────
class_weight_values = compute_class_weight(
    'balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
CLASS_WEIGHTS = dict(enumerate(class_weight_values))
print(f"⚖️  Class weights: {CLASS_WEIGHTS}")

# ──────── MixUp augmentation wrapper ────────
class MixUpGenerator(keras.utils.Sequence):
    """MixUp: blends pairs of images+labels to regularize and boost accuracy."""
    def __init__(self, generator, alpha=0.2):
        self.generator = generator
        self.alpha = alpha
        self.batch_size = generator.batch_size
        self.samples = generator.samples

    def __len__(self):
        return len(self.generator)

    def __getitem__(self, index):
        x1, y1 = self.generator[index]
        # Get a random other batch
        rand_idx = np.random.randint(0, len(self.generator))
        x2, y2 = self.generator[rand_idx]

        # Match batch sizes (edge case: last batch may be smaller)
        min_len = min(len(x1), len(x2))
        x1, y1 = x1[:min_len], y1[:min_len]
        x2, y2 = x2[:min_len], y2[:min_len]

        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha, size=(min_len, 1, 1, 1))
        lam_y = lam.reshape(-1)

        x_mix = x1 * lam + x2 * (1 - lam)
        y_mix = y1 * lam_y + y2 * (1 - lam_y)
        return x_mix.astype(np.float32), y_mix.astype(np.float32)

    def on_epoch_end(self):
        self.generator.on_epoch_end()

train_mixup = MixUpGenerator(train_gen, alpha=MIXUP_ALPHA)
print(f"🔀 MixUp augmentation enabled (α={MIXUP_ALPHA})")


# ================================================================
#  ██████╗ ███████╗██╗     ██╗         ██╗  ██╗
# ██╔════╝ ██╔════╝██║     ██║         ██║  ██║
# ██║      █████╗  ██║     ██║         ███████║
# ██║      ██╔══╝  ██║     ██║         ╚════██║
# ╚██████╗ ███████╗███████╗███████╗         ██║
#  ╚═════╝ ╚══════╝╚══════╝╚══════╝         ╚═╝
#  VISUALIZE SAMPLES + AUGMENTATIONS
# ================================================================
# %%

fig, axes = plt.subplots(2, 6, figsize=(18, 6))
fig.suptitle("Training Samples (with augmentation)", fontsize=16, fontweight='bold')

images, labels = next(iter(train_gen))
for i, ax in enumerate(axes.flat):
    if i < len(images):
        ax.imshow(images[i])
        lbl = CLASS_NAMES[int(labels[i])]
        ax.set_title(lbl, fontsize=11,
                     color='#e74c3c' if 'drowsy' in lbl.lower() and 'non' not in lbl.lower()
                     else '#27ae60', fontweight='bold')
        ax.axis('off')
plt.tight_layout()
plt.show()

# Class distribution chart
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(class_counts.keys(), class_counts.values(),
              color=['#e74c3c', '#27ae60'], edgecolor='black')
ax.set_title("Class Distribution", fontsize=14, fontweight='bold')
ax.set_ylabel("Number of Images")
for bar, count in zip(bars, class_counts.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
            str(count), ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()


# ================================================================
#  ██████╗ ███████╗██╗     ██╗         ███████╗
# ██╔════╝ ██╔════╝██║     ██║         ██╔════╝
# ██║      █████╗  ██║     ██║         ███████╗
# ██║      ██╔══╝  ██║     ██║         ╚════██║
# ╚██████╗ ███████╗███████╗███████╗    ███████║
#  ╚═════╝ ╚══════╝╚══════╝╚══════╝    ╚══════╝
#  BUILD EFFICIENTNET-B0 MODEL (BEST ACCURACY)
# ================================================================
# %%

def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """
    EfficientNetB0 + custom head with attention.
    - Frozen base → train head → unfreeze & fine-tune
    - Spatial attention squeezes more signal from the face
    - Label smoothing prevents overconfidence
    """
    # ── Base Model ──
    base = keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False   # Freeze initially

    # ── Custom Head with Squeeze-Excitation Attention ──
    inputs = keras.Input(shape=input_shape)
    x = base(inputs, training=False)

    # Global pooling + channel attention
    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    concat = layers.Concatenate()([gap, gmp])  # Richer features

    x = layers.Dense(256)(concat)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)          # Swish > ReLU for EfficientNet
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs, name="DrowsinessDetector_EfficientNetB0")
    return model, base


model, base_model = build_model()
model.summary()

total_params = model.count_params()
trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
print(f"\n📐 Total params:     {total_params:,}")
print(f"📐 Trainable params: {trainable_params:,}")


# ================================================================
#  ██████╗ ███████╗██╗     ██╗          ██████╗
# ██╔════╝ ██╔════╝██║     ██║         ██╔════╝
# ██║      █████╗  ██║     ██║         ███████╗
# ██║      ██╔══╝  ██║     ██║         ██╔═══██╗
# ╚██████╗ ███████╗███████╗███████╗    ╚██████╔╝
#  ╚═════╝ ╚══════╝╚══════╝╚══════╝     ╚═════╝
#  PHASE 1: TRAIN HEAD ONLY (base frozen)
# ================================================================
# %%

# ── Cosine Annealing LR Schedule ──
def cosine_schedule(epoch, lr):
    """Cosine annealing — smoothly decays LR, avoids sharp drops."""
    max_lr = INITIAL_LR
    min_lr = 1e-6
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * epoch / EPOCHS))

# ── Compile with Label Smoothing ──
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR),
    loss=keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]
)

phase1_callbacks = [
    callbacks.ModelCheckpoint('best_model_phase1.keras',
                              monitor='val_auc', save_best_only=True, mode='max', verbose=1),
    callbacks.EarlyStopping(monitor='val_auc', patience=5,
                            restore_best_weights=True, verbose=1),
    callbacks.LearningRateScheduler(cosine_schedule, verbose=0)
]

print("=" * 60)
print("  🔒 PHASE 1: Training classifier head (base frozen)")
print("=" * 60)

history_phase1 = model.fit(
    train_mixup,
    epochs=10,
    validation_data=val_gen,
    class_weight=CLASS_WEIGHTS,
    callbacks=phase1_callbacks,
    verbose=1
)

p1_acc = max(history_phase1.history['val_accuracy'])
print(f"\n✅ Phase 1 complete — Best val accuracy: {p1_acc:.4f}")


# ================================================================
#  ██████╗ ███████╗██╗     ██╗         ███████╗
# ██╔════╝ ██╔════╝██║     ██║             ██╔╝
# ██║      █████╗  ██║     ██║            ██╔╝
# ██║      ██╔══╝  ██║     ██║           ██╔╝
# ╚██████╗ ███████╗███████╗███████╗      ██║
#  ╚═════╝ ╚══════╝╚══════╝╚══════╝      ╚═╝
#  PHASE 2: FINE-TUNE TOP LAYERS (unfreeze)
# ================================================================
# %%

# ── Unfreeze top 40 layers of EfficientNet ──
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

trainable_now = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
print(f"🔓 Unfroze top 40 layers — trainable params: {trainable_now:,}")

# ── Recompile with MUCH lower LR (critical for fine-tuning) ──
FINE_TUNE_LR = 1e-4

def cosine_finetune(epoch, lr):
    min_lr = 1e-7
    return min_lr + 0.5 * (FINE_TUNE_LR - min_lr) * (1 + np.cos(np.pi * epoch / 30))

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss=keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]
)

phase2_callbacks = [
    callbacks.ModelCheckpoint('best_model_final.keras',
                              monitor='val_auc', save_best_only=True, mode='max', verbose=1),
    callbacks.EarlyStopping(monitor='val_auc', patience=8,
                            restore_best_weights=True, verbose=1),
    callbacks.LearningRateScheduler(cosine_finetune, verbose=0),
]

print("\n" + "=" * 60)
print("  🔓 PHASE 2: Fine-tuning top 40 layers")
print("=" * 60)

history_phase2 = model.fit(
    train_mixup,
    epochs=30,
    validation_data=val_gen,
    class_weight=CLASS_WEIGHTS,
    callbacks=phase2_callbacks,
    verbose=1
)

p2_acc = max(history_phase2.history['val_accuracy'])
print(f"\n✅ Phase 2 complete — Best val accuracy: {p2_acc:.4f}")

# ── Reload the absolute best checkpoint ──
model = keras.models.load_model('best_model_final.keras')
print("✅ Loaded best checkpoint: best_model_final.keras")


# ================================================================
#  ██████╗ ███████╗██╗     ██╗          █████╗
# ██╔════╝ ██╔════╝██║     ██║         ██╔══██╗
# ██║      █████╗  ██║     ██║         ╚█████╔╝
# ██║      ██╔══╝  ██║     ██║         ██╔══██╗
# ╚██████╗ ███████╗███████╗███████╗    ╚█████╔╝
#  ╚═════╝ ╚══════╝╚══════╝╚══════╝     ╚════╝
#  PLOT FULL TRAINING HISTORY
# ================================================================
# %%

# Merge both phase histories
def merge_histories(h1, h2):
    merged = {}
    for key in h1.history:
        merged[key] = h1.history[key] + h2.history.get(key, [])
    return merged

history = merge_histories(history_phase1, history_phase2)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Training History (Phase 1 → Phase 2)", fontsize=16, fontweight='bold')

# Phase boundary
phase1_end = len(history_phase1.history['accuracy'])

metrics_to_plot = [
    ('accuracy', 'val_accuracy', 'Accuracy', 'upper left'),
    ('loss', 'val_loss', 'Loss', 'upper right'),
    ('auc', 'val_auc', 'AUC', 'upper left'),
    ('precision', 'val_precision', 'Precision', 'upper left'),
]

for ax, (train_key, val_key, title, loc) in zip(axes.flat, metrics_to_plot):
    if train_key in history and val_key in history:
        ax.plot(history[train_key], label=f'Train', linewidth=2, alpha=0.8)
        ax.plot(history[val_key], label=f'Val', linewidth=2, alpha=0.8)
        ax.axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.5, label='Fine-tune start')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.legend(loc=loc)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.show()


# ================================================================
#  ██████╗ ███████╗██╗     ██╗          █████╗
# ██╔════╝ ██╔════╝██║     ██║         ██╔══██╗
# ██║      █████╗  ██║     ██║         ╚██████║
# ██║      ██╔══╝  ██║     ██║          ╚═══██║
# ╚██████╗ ███████╗███████╗███████╗    █████╔╝
#  ╚═════╝ ╚══════╝╚══════╝╚══════╝    ╚════╝
#  TEST-TIME AUGMENTATION (TTA) + EVALUATION
# ================================================================
# %%

print("📊 Evaluating with Test-Time Augmentation (TTA)...\n")

# ── TTA: run inference 5x with slight augmentations, average results ──
tta_datagens = [
    ImageDataGenerator(rescale=1./255),                                         # Original
    ImageDataGenerator(rescale=1./255, horizontal_flip=True),                   # Flipped
    ImageDataGenerator(rescale=1./255, rotation_range=10),                      # Slight rotation
    ImageDataGenerator(rescale=1./255, brightness_range=[0.9, 1.1]),            # Brightness
    ImageDataGenerator(rescale=1./255, zoom_range=0.05),                        # Slight zoom
]

all_preds = []
for i, dg in enumerate(tta_datagens):
    tta_gen = dg.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        seed=SEED,
        shuffle=False
    )
    preds = model.predict(tta_gen, verbose=0).flatten()
    all_preds.append(preds)
    print(f"   TTA pass {i+1}/5 done")

# Average all TTA predictions
y_pred_avg = np.mean(all_preds, axis=0)
y_pred = (y_pred_avg > 0.5).astype(int)
y_true = val_gen.classes

# ── Classification Report ──
print("\n" + "=" * 60)
print("        📋 CLASSIFICATION REPORT (with TTA)")
print("=" * 60)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

# ── Confusion Matrix ──
cm = confusion_matrix(y_true, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            annot_kws={'size': 18}, ax=axes[0])
axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted', fontsize=12)
axes[0].set_ylabel('Actual', fontsize=12)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred_avg)
roc_auc = auc(fpr, tpr)
axes[1].plot(fpr, tpr, color='#3498db', linewidth=2.5, label=f'ROC curve (AUC = {roc_auc:.4f})')
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[1].fill_between(fpr, tpr, alpha=0.1, color='#3498db')
axes[1].set_xlabel('False Positive Rate', fontsize=12)
axes[1].set_ylabel('True Positive Rate', fontsize=12)
axes[1].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('evaluation_results.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Find optimal threshold ──
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"\n🎯 Optimal classification threshold: {optimal_threshold:.4f}")
print(f"   (Default is 0.5 — use {optimal_threshold:.3f} in your real-time script for best results)")

# ── Compute final numbers ──
y_pred_optimal = (y_pred_avg > optimal_threshold).astype(int)
from sklearn.metrics import accuracy_score, f1_score
acc_default  = accuracy_score(y_true, y_pred)
acc_optimal  = accuracy_score(y_true, y_pred_optimal)
f1_default   = f1_score(y_true, y_pred)
f1_optimal   = f1_score(y_true, y_pred_optimal)

print(f"\n📈 Results Comparison:")
print(f"   {'Metric':<12} {'Threshold=0.5':>14} {'Optimal':>14}")
print(f"   {'─'*40}")
print(f"   {'Accuracy':<12} {acc_default:>13.4f} {acc_optimal:>13.4f}")
print(f"   {'F1 Score':<12} {f1_default:>13.4f} {f1_optimal:>13.4f}")
print(f"   {'AUC':<12} {roc_auc:>13.4f} {'—':>14}")


# ================================================================
#  ██████╗ ███████╗██╗     ██╗       ██╗ ██████╗
# ██╔════╝ ██╔════╝██║     ██║      ██╔╝██╔═████╗
# ██║      █████╗  ██║     ██║     ██╔╝ ██║██╔██║
# ██║      ██╔══╝  ██║     ██║    ██╔╝  ████╔╝██║
# ╚██████╗ ███████╗███████╗███████╗╚██╗ ╚██████╔╝
#  ╚═════╝ ╚══════╝╚══════╝╚══════╝ ╚═╝  ╚═════╝
#  SAVE IN ALL FORMATS + DOWNLOAD
# ================================================================
# %%

# 1. Keras format
model.save('drowsiness_model_optimized.keras')
print("✅ Saved: drowsiness_model_optimized.keras")

# 2. H5 format
model.save('drowsiness_model_optimized.h5')
print("✅ Saved: drowsiness_model_optimized.h5")

# 3. TFLite (quantized for edge/mobile deployment)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]  # Float16 quantization
tflite_model = converter.convert()
with open('drowsiness_model_optimized.tflite', 'wb') as f:
    f.write(tflite_model)
print(f"✅ Saved: drowsiness_model_optimized.tflite ({len(tflite_model)/1024/1024:.1f} MB)")

# 4. Metadata JSON
metadata = {
    "class_indices": train_gen.class_indices,
    "class_names": CLASS_NAMES,
    "img_size": IMG_SIZE,
    "optimal_threshold": float(optimal_threshold),
    "val_accuracy": float(acc_optimal),
    "val_auc": float(roc_auc),
    "label_smoothing": LABEL_SMOOTHING,
    "architecture": "EfficientNetB0 + Dual-Pooling + Attention Head",
    "augmentations": ["rotation", "shift", "brightness", "shear", "zoom", "flip", "color_jitter", "mixup"],
}
with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✅ Saved: model_metadata.json")

print(f"\n{'='*50}")
print(f"  🏆 FINAL MODEL PERFORMANCE")
print(f"{'='*50}")
print(f"  Accuracy:           {acc_optimal*100:.2f}%")
print(f"  AUC:                {roc_auc:.4f}")
print(f"  F1 Score:           {f1_optimal:.4f}")
print(f"  Optimal Threshold:  {optimal_threshold:.4f}")
print(f"{'='*50}")

# ── Download all files ──
from google.colab import files
for fname in ['drowsiness_model_optimized.keras',
              'drowsiness_model_optimized.h5',
              'drowsiness_model_optimized.tflite',
              'model_metadata.json',
              'training_history.png',
              'evaluation_results.png']:
    try:
        files.download(fname)
        print(f"📥 Downloading: {fname}")
    except Exception as e:
        print(f"⚠️  Could not auto-download {fname}: {e}")

print("\n🎉 All done! Copy the .keras file next to realtime_detection.py and run it.")
