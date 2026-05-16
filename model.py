import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers


def conv_block(x, filters, kernel_size=5, dropout_rate=0.1, l2=1e-4):
    """Conv → BatchNorm → ReLU → MaxPool → Dropout"""
    x = layers.Conv2D(
        filters, kernel_size,
        padding='same',
        use_bias=False,                          # BatchNorm handles bias
        kernel_regularizer=regularizers.l2(l2)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout_rate)(x)
    return x


def build_model(num_classes=43, l2=1e-4):
    """
    CNN with 4 conv blocks + multi-scale skip connections.

    Architecture:
        Input (32x32x1)
        → Conv Block 1:  32 filters, 5x5, 10% dropout  → (16x16x32)
        → Conv Block 2:  64 filters, 5x5, 20% dropout  → (8x8x64)
        → Conv Block 3: 128 filters, 5x5, 30% dropout  → (4x4x128)
        → Conv Block 4: 256 filters, 3x3, 40% dropout  → (2x2x256)
        → Multi-scale merge (x1 + x2 + x3 + x4 flattened)
        → Dense 1024 → 50% Dropout
        → Output: num_classes (Softmax)
    """
    inp = layers.Input(shape=(32, 32, 1), name='input')

    # --- Conv blocks ---
    x1 = conv_block(inp, filters=32,  kernel_size=5, dropout_rate=0.1, l2=l2)
    x2 = conv_block(x1,  filters=64,  kernel_size=5, dropout_rate=0.2, l2=l2)
    x3 = conv_block(x2,  filters=128, kernel_size=5, dropout_rate=0.3, l2=l2)
    x4 = conv_block(x3,  filters=256, kernel_size=3, dropout_rate=0.4, l2=l2)

    # --- Multi-scale skip connections ---
    # Pool earlier layers down to match x4's spatial size (2x2)
    x1_flat = layers.Flatten()(layers.MaxPooling2D(8)(x1))   # 16x16 → 2x2
    x2_flat = layers.Flatten()(layers.MaxPooling2D(4)(x2))   # 8x8   → 2x2
    x3_flat = layers.Flatten()(layers.MaxPooling2D(2)(x3))   # 4x4   → 2x2
    x4_flat = layers.Flatten()(x4)                           # 2x2   → flat

    merged = layers.Concatenate(name='multi_scale_merge')(
        [x1_flat, x2_flat, x3_flat, x4_flat]
    )

    # --- Classifier head ---
    fc = layers.Dense(1024, use_bias=False)(merged)
    fc = layers.BatchNormalization()(fc)
    fc = layers.Activation('relu')(fc)
    fc = layers.Dropout(0.5)(fc)

    out = layers.Dense(num_classes, activation='softmax', name='output')(fc)

    return Model(inp, out, name='TrafficSignCNN')


def get_callbacks(monitor='val_loss', patience=100, model_path='best_model.keras'):
    """Standard callbacks: EarlyStopping + ModelCheckpoint + ReduceLROnPlateau"""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor=monitor,
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=20,
            min_lr=1e-6,
            verbose=1
        ),
    ]


if __name__ == '__main__':
    model = build_model()
    model.summary()

    total     = model.count_params()
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"\nTotal params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
