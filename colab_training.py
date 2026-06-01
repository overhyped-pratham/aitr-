# =============================================================================
#  DRIVER DROWSINESS DETECTION — GOOGLE COLAB TRAINING NOTEBOOK
#  Dataset: ismailnasri20/driver-drowsiness-dataset-ddd
#  Copy each cell (separated by # %%) into a Google Colab notebook
# =============================================================================

# %% [markdown]
# # 🚗 Driver Drowsiness Detection — Model Training
# **Dataset:** DDD (Driver Drowsiness Dataset) — ~41,790 images (227×227)
# **Classes:** Drowsy | Non-Drowsy
# **Architecture:** Custom CNN + optional MobileNetV2 Transfer Learning

# ============================================================
# CELL 1 — Install Dependencies
# ============================================================
# %%
!pip install -q kagglehub tensorflow matplotlib seaborn scikit-learn

# ============================================================
# CELL 2 — Download Dataset from Kaggle
# ============================================================
# %%
import kagglehub
import os

# Download the dataset
path = kagglehub.dataset_download("ismailnasri20/driver-drowsiness-dataset-ddd")
print("✅ Dataset downloaded to:", path)

# Explore the folder structure
for root, dirs, files in os.walk(path):
    level = root.replace(path, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}📁 {os.path.basename(root)}/  ({len(files)} files)")
    if level >= 2:  # Don't go too deep
        break

# ============================================================
# CELL 3 — Imports & Configuration
# ============================================================
# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ---------- CONFIGURATION ----------
IMG_SIZE      = 128          # Resize to 128x128 (faster training, good accuracy)
BATCH_SIZE    = 32
EPOCHS        = 25
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
RANDOM_SEED   = 42

# Auto-detect the dataset root (find folder containing 'Drowsy' subfolder)
DATASET_DIR = None
for root, dirs, files in os.walk(path):
    lower_dirs = [d.lower() for d in dirs]
    if 'drowsy' in lower_dirs or 'non drowsy' in lower_dirs:
        DATASET_DIR = root
        break
    # Also check for common alternate structures
    if 'train' in lower_dirs:
        DATASET_DIR = os.path.join(root, 'train')
        break

if DATASET_DIR is None:
    DATASET_DIR = path  # Fallback

print(f"📂 Using dataset directory: {DATASET_DIR}")
print(f"📁 Contents: {os.listdir(DATASET_DIR)}")

# ============================================================
# CELL 4 — Load & Prepare Data (with Augmentation)
# ============================================================
# %%
# Training data generator WITH augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=VALIDATION_SPLIT,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    zoom_range=0.1,
    fill_mode='nearest'
)

# Validation data generator — NO augmentation, only rescale
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=VALIDATION_SPLIT
)

# Load training set
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    seed=RANDOM_SEED,
    shuffle=True
)

# Load validation set
val_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    seed=RANDOM_SEED,
    shuffle=False
)

# Print class mapping
print("\n🏷️  Class Indices:", train_generator.class_indices)
print(f"📊 Training samples:   {train_generator.samples}")
print(f"📊 Validation samples: {val_generator.samples}")

# ============================================================
# CELL 5 — Visualize Sample Images
# ============================================================
# %%
class_names = list(train_generator.class_indices.keys())

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle("Sample Images from Dataset", fontsize=16, fontweight='bold')

images, labels = next(train_generator)
for i, ax in enumerate(axes.flat):
    if i < len(images):
        ax.imshow(images[i])
        ax.set_title(class_names[int(labels[i])], fontsize=11,
                     color='red' if class_names[int(labels[i])].lower() == 'drowsy' else 'green')
        ax.axis('off')

plt.tight_layout()
plt.show()

# ============================================================
# CELL 6 — Build Custom CNN Model
# ============================================================
# %%
def build_custom_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """
    A custom CNN optimized for drowsiness detection.
    ~1.5M parameters — fast to train, accurate, easy to deploy.
    """
    model = keras.Sequential([
        # --- Block 1 ---
        layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # --- Block 2 ---
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # --- Block 3 ---
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # --- Block 4 ---
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # --- Classifier Head ---
        layers.GlobalAveragePooling2D(),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')   # Binary output
    ], name="DrowsinessDetector_CNN")

    return model


model = build_custom_cnn()
model.summary()

# ============================================================
# CELL 7 — Compile the Model
# ============================================================
# %%
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]
)

print("✅ Model compiled successfully")

# ============================================================
# CELL 8 — Set up Training Callbacks
# ============================================================
# %%
my_callbacks = [
    # Save the best model based on validation accuracy
    callbacks.ModelCheckpoint(
        'best_drowsiness_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    # Stop early if no improvement for 7 epochs
    callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate when validation loss plateaus
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

print("✅ Callbacks configured")

# ============================================================
# CELL 9 — Train the Model 🚀
# ============================================================
# %%
print("🚀 Starting training...\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=my_callbacks,
    verbose=1
)

print("\n✅ Training complete!")

# ============================================================
# CELL 10 — Plot Training History
# ============================================================
# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# AUC
axes[2].plot(history.history['auc'], label='Train AUC', linewidth=2)
axes[2].plot(history.history['val_auc'], label='Val AUC', linewidth=2)
axes[2].set_title('Model AUC', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('AUC')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# CELL 11 — Evaluate on Validation Set
# ============================================================
# %%
print("📊 Evaluating on validation set...\n")

# Get predictions
val_generator.reset()
y_pred_probs = model.predict(val_generator, verbose=1)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()
y_true = val_generator.classes

# Classification Report
print("\n" + "=" * 55)
print("            CLASSIFICATION REPORT")
print("=" * 55)
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={'size': 16})
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Predicted', fontsize=13)
plt.ylabel('Actual', fontsize=13)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# Print final metrics
val_loss, val_acc, val_prec, val_rec, val_auc = model.evaluate(val_generator, verbose=0)
print(f"\n🎯 Final Validation Metrics:")
print(f"   Accuracy:  {val_acc:.4f}  ({val_acc*100:.1f}%)")
print(f"   Precision: {val_prec:.4f}")
print(f"   Recall:    {val_rec:.4f}")
print(f"   AUC:       {val_auc:.4f}")
print(f"   Loss:      {val_loss:.4f}")

# ============================================================
# CELL 12 — Save Model in Multiple Formats
# ============================================================
# %%
# 1. Save as Keras format (recommended)
model.save('drowsiness_model.keras')
print("✅ Saved: drowsiness_model.keras")

# 2. Save as H5 format (legacy compatibility)
model.save('drowsiness_model.h5')
print("✅ Saved: drowsiness_model.h5")

# 3. Save as TFLite (for mobile / Raspberry Pi deployment)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('drowsiness_model.tflite', 'wb') as f:
    f.write(tflite_model)
print(f"✅ Saved: drowsiness_model.tflite ({len(tflite_model) / 1024 / 1024:.1f} MB)")

# 4. Save class names for inference
import json
with open('class_names.json', 'w') as f:
    json.dump(train_generator.class_indices, f)
print("✅ Saved: class_names.json")

print("\n📦 All models saved! Download them from the Colab file browser (left sidebar).")

# ============================================================
# CELL 13 — Download Models to Your PC (Run in Colab)
# ============================================================
# %%
from google.colab import files

print("📥 Downloading model files...\n")
files.download('drowsiness_model.keras')
files.download('drowsiness_model.h5')
files.download('drowsiness_model.tflite')
files.download('class_names.json')
files.download('training_history.png')
files.download('confusion_matrix.png')

# ============================================================
# CELL 14 (BONUS) — Transfer Learning with MobileNetV2
# ============================================================
# %% [markdown]
# ## 🔄 BONUS: MobileNetV2 Transfer Learning
# Run this cell INSTEAD of Cells 6-9 if you want higher accuracy.
# MobileNetV2 is pre-trained on ImageNet — great for face images.

# %%
def build_mobilenet_model(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """MobileNetV2-based model — higher accuracy, slightly larger."""

    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Freeze the base model initially
    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ], name="DrowsinessDetector_MobileNetV2")

    return model, base_model


mobilenet_model, base = build_mobilenet_model()
mobilenet_model.summary()

# --- Phase 1: Train classifier head only ---
mobilenet_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n🔒 Phase 1: Training classifier head (base frozen)...")
mobilenet_model.fit(train_generator, epochs=5, validation_data=val_generator, verbose=1)

# --- Phase 2: Fine-tune top layers ---
base.trainable = True
for layer in base.layers[:-30]:  # Freeze all but last 30 layers
    layer.trainable = False

mobilenet_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),  # Lower LR for fine-tuning
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n🔓 Phase 2: Fine-tuning top 30 layers...")
mobilenet_model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[
        callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    ],
    verbose=1
)

# Save
mobilenet_model.save('drowsiness_mobilenet.keras')
print("\n✅ MobileNetV2 model saved!")
