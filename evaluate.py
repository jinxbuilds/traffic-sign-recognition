import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data_loader import load_data
from preprocess import preprocess

# --- Load test data ---
_, _, (X_test, y_test) = load_data('train.p', 'valid.p', 'test.p')
X_test, y_test_oh = preprocess(X_test, y_test)

# --- Load model ---
model = tf.keras.models.load_model('stage2.keras')

# --- Accuracy ---
loss, acc = model.evaluate(X_test, y_test_oh, verbose=0)
print(f"Test accuracy: {acc * 100:.2f}%")

# --- Error analysis ---
y_pred = model.predict(X_test).argmax(axis=1)
y_true = y_test_oh.argmax(axis=1)
errors = np.where(y_pred != y_true)[0]
print(f"Errors: {len(errors)} / {len(y_true)}")

# --- Plot misclassified images ---
fig, axes = plt.subplots(5, 10, figsize=(20, 10))
for ax, idx in zip(axes.flat, errors[:50]):
    ax.imshow(X_test[idx].squeeze(), cmap='gray')
    ax.set_title(f"T:{y_true[idx]}\nP:{y_pred[idx]}", fontsize=7)
    ax.axis('off')
plt.suptitle("Misclassified Samples", fontsize=14)
plt.tight_layout()
plt.savefig('errors.png', dpi=150)
plt.show()

# --- Plot conv1 filters ---
weights = model.layers[1].get_weights()[0]  # (5, 5, 1, 32)
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for i, ax in enumerate(axes.flat):
    f = weights[:, :, 0, i]
    ax.imshow(f, cmap='gray')
    ax.axis('off')
plt.suptitle("Conv1 Filters", fontsize=14)
plt.tight_layout()
plt.savefig('filters.png', dpi=150)
plt.show()