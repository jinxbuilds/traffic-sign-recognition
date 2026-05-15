import os, csv, pickle
import numpy as np
from skimage.io import imread
from skimage.transform import resize

def load_training(root, img_size=32):
    X, y = [], []
    for class_id in range(43):
        class_dir = os.path.join(root, f"{class_id:05d}")
        csv_file  = os.path.join(class_dir, f"GT-{class_id:05d}.csv")
        with open(csv_file) as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                img_path = os.path.join(class_dir, row['Filename'])
                img = imread(img_path)
                img = resize(img, (img_size, img_size), anti_aliasing=True)
                img = (img * 255).astype(np.uint8)
                X.append(img)
                y.append(int(row['ClassId']))
    return np.array(X), np.array(y)

def load_test(images_dir, gt_csv, img_size=32):
    X, y = [], []
    with open(gt_csv) as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            img_path = os.path.join(images_dir, row['Filename'])
            img = imread(img_path)
            img = resize(img, (img_size, img_size), anti_aliasing=True)
            img = (img * 255).astype(np.uint8)
            X.append(img)
            y.append(int(row['ClassId']))
    return np.array(X), np.array(y)

# --- Paths (update these) ---
TRAIN_DIR = 'dataset/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images'
TEST_DIR  = 'dataset/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images'
TEST_GT   = 'dataset/GTSRB_Final_Test_GT/GT-final_test.csv'

print("Loading training data...")
X_train, y_train = load_training(TRAIN_DIR)

# Split 80/20 for train/valid
split = int(len(X_train) * 0.8)
X_valid, y_valid = X_train[split:], y_train[split:]
X_train, y_train = X_train[:split], y_train[:split]

print("Loading test data...")
X_test, y_test = load_test(TEST_DIR, TEST_GT)

# Save as pickle
for name, X, y in [('train', X_train, y_train),
                   ('valid', X_valid, y_valid),
                   ('test',  X_test,  y_test)]:
    with open(f'dataset/{name}.p', 'wb') as f:
        pickle.dump({'features': X, 'labels': y}, f)
    print(f"Saved dataset/{name}.p — {X.shape}")

print("Done!")