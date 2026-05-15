import os
import cv2
import numpy as np
import tensorflow as tf

from preprocess import preprocess


# =========================================================
# Load model
# =========================================================

model = tf.keras.models.load_model('best_model.keras')


# =========================================================
# Class names
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
# Dataset test folder
# =========================================================

TEST_FOLDER = r'dataset/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images'


# =========================================================
# Read test CSV labels
# =========================================================

csv_path = r'dataset/GTSRB_Final_Test_GT/GT-final_test.csv'

ground_truth = {}

with open(csv_path, 'r') as f:

    lines = f.readlines()[1:]

    for line in lines:

        parts = line.strip().split(';')

        filename = parts[0]

        class_id = int(parts[-1])

        ground_truth[filename] = class_id


# =========================================================
# Predict multiple images
# =========================================================

correct = 0
total = 0

files = list(ground_truth.keys())[:100]

for filename in files:

    path = os.path.join(TEST_FOLDER, filename)

    img = cv2.imread(path)

    if img is None:
        continue

    img = cv2.resize(img, (32, 32))

    X, _ = preprocess(np.array([img]))

    pred = model.predict(X, verbose=0)

    predicted_class = int(np.argmax(pred))

    confidence = float(np.max(pred))

    actual_class = ground_truth[filename]

    is_correct = predicted_class == actual_class

    if is_correct:
        correct += 1

    total += 1

    print("\n--------------------------------")

    print(f"Image      : {filename}")

    print(f"Actual     : {sign_names[actual_class]}")

    print(f"Predicted  : {sign_names[predicted_class]}")

    print(f"Confidence : {confidence * 100:.2f}%")

    print(f"Correct    : {is_correct}")

    print("--------------------------------")


# =========================================================
# Final accuracy
# =========================================================

accuracy = (correct / total) * 100

print("\n================================")
print("FINAL RESULTS")
print("================================")

print(f"Images Tested : {total}")

print(f"Correct       : {correct}")

print(f"Accuracy      : {accuracy:.2f}%")

print("================================")