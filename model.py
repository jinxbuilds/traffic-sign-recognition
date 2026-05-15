import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

def build_model(num_classes=43, l2=1e-4):
    inp = layers.Input(shape=(32, 32, 1))

    # Conv block 1
    x1 = layers.Conv2D(32, 5, padding='same', activation='relu',
                        kernel_regularizer=regularizers.l2(l2))(inp)
    x1 = layers.MaxPooling2D(2)(x1)
    x1 = layers.Dropout(0.1)(x1)

    # Conv block 2
    x2 = layers.Conv2D(64, 5, padding='same', activation='relu',
                        kernel_regularizer=regularizers.l2(l2))(x1)
    x2 = layers.MaxPooling2D(2)(x2)
    x2 = layers.Dropout(0.2)(x2)

    # Conv block 3
    x3 = layers.Conv2D(128, 5, padding='same', activation='relu',
                        kernel_regularizer=regularizers.l2(l2))(x2)
    x3 = layers.MaxPooling2D(2)(x3)
    x3 = layers.Dropout(0.3)(x3)

    # Multi-scale skip connections
    x1_flat = layers.Flatten()(layers.MaxPooling2D(4)(x1))
    x2_flat = layers.Flatten()(layers.MaxPooling2D(2)(x2))
    x3_flat = layers.Flatten()(x3)

    merged = layers.Concatenate()([x1_flat, x2_flat, x3_flat])

    fc = layers.Dense(1024, activation='relu')(merged)
    fc = layers.Dropout(0.5)(fc)
    out = layers.Dense(num_classes, activation='softmax')(fc)

    return Model(inp, out)