import cv2
import os
import numpy as np
import tensorflow as tf
from config import DATASET_DIR
from preprocess import preprocess


# =========================================================
# Load trained model
# =========================================================

model = tf.keras.models.load_model('stage2.keras')


# =========================================================
# Traffic sign labels
# =========================================================

sign_names = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 tons",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End no passing vehicle > 3.5 tons"
}


# =========================================================
# Load image
# =========================================================

SAMPLE_IMAGE = os.path.join(DATASET_DIR, 'GTSRB_Final_Test_Images', 'GTSRB', 'Final_Test', 'Images', '00016.ppm')

img = cv2.imread(IMAGE_PATH)

if img is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )


# =========================================================
# Resize image
# =========================================================

img_resized = cv2.resize(
    img,
    (32, 32)
)


# =========================================================
# Preprocess
# =========================================================

X, _ = preprocess(
    np.array([img_resized])
)


# =========================================================
# Predict
# =========================================================

pred = model.predict(X, verbose=0)

label = int(np.argmax(pred))

confidence = float(np.max(pred))


# =========================================================
# Results
# =========================================================

print("\n==============================")
print("Traffic Sign Prediction")
print("==============================")

print(f"Class ID   : {label}")

print(f"Sign Name  : {sign_names[label]}")

print(f"Confidence : {confidence * 100:.2f}%")

print("==============================\n")


# =========================================================
# Display image
# =========================================================

display = cv2.resize(img, (400, 400))

cv2.putText(
    display,
    sign_names[label],
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)

cv2.putText(
    display,
    f"{confidence * 100:.2f}%",
    (10, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)

cv2.imshow("Prediction", display)

cv2.waitKey(0)

cv2.destroyAllWindows()
