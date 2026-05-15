import numpy as np
import tensorflow as tf

from data_loader import load_data
from preprocess import preprocess, one_hot
from augment import flip_extend, make_extended, make_balanced
from model import build_model


# --- Load dataset ---

(X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_data(
    'dataset/train.p',
    'dataset/valid.p',
    'dataset/test.p'
)

print(f"Train: {X_train.shape}")
print(f"Valid: {X_valid.shape}")
print(f"Test : {X_test.shape}")


# --- Preprocess ---

# keep train labels as integers for augmentation
X_train, _ = preprocess(X_train)

# validation + test can be one-hot encoded immediately
X_valid, y_valid_oh = preprocess(X_valid, y_valid)
X_test, y_test_oh = preprocess(X_test, y_test)


# --- Flip augmentation ---

print("\nApplying flip augmentation...")

X_flipped, y_flipped = flip_extend(X_train, y_train)

print(f"Flipped dataset: {X_flipped.shape}")


# --- Debug class distribution ---

print("\nClass distribution after flipping:")

for c in range(43):

    count = np.sum(y_flipped == c)

    print(f"Class {c}: {count}")

    if count == 0:
        print(f"WARNING -> Class {c} is EMPTY")


# --- Extended dataset ---

print("\nGenerating extended dataset...")

X_ext, y_ext = make_extended(
    X_flipped,
    y_flipped,
    multiplier=5
)

print(f"Extended dataset: {X_ext.shape}")


# --- Balanced dataset ---

print("\nGenerating balanced dataset...")

X_bal, y_bal = make_balanced(
    X_flipped,
    y_flipped,
    per_class=5000
)

print(f"Balanced dataset: {X_bal.shape}")


# --- One-hot encode augmented labels ---

y_ext_oh = one_hot(y_ext)
y_bal_oh = one_hot(y_bal)


# --- Callbacks ---

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True,
    verbose=1
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)


# --- Stage 1: Pre-training ---

print("\n=== Stage 1: Pre-training ===")

model = build_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_ext,
    y_ext_oh,
    validation_data=(X_valid, y_valid_oh),
    batch_size=128,
    epochs=50,
    callbacks=[early_stop, checkpoint],
    shuffle=True
)

model.save('stage1.keras')

print("\nStage 1 complete.")


# --- Stage 2: Fine-tuning ---

print("\n=== Stage 2: Fine-tuning ===")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_bal,
    y_bal_oh,
    validation_data=(X_valid, y_valid_oh),
    batch_size=128,
    epochs=50,
    callbacks=[early_stop, checkpoint],
    shuffle=True
)

model.save('stage2.keras')

print("\nStage 2 complete.")


# --- Final evaluation ---

print("\nEvaluating on test set...")

loss, acc = model.evaluate(X_test, y_test_oh)

print(f"\nTest Accuracy: {acc * 100:.2f}%")