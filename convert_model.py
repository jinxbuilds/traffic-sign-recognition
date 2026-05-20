import tensorflow as tf

# Load current model
model = tf.keras.models.load_model("model/stage2.keras")

# Save in H5 format
model.save("model/stage2.h5")

print("Model converted successfully")