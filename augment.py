import numpy as np
import random
from skimage.transform import rotate, warp, ProjectiveTransform
from config import NUM_CLASSES, MULTIPLIER, PER_CLASS


# =========================================================
# Flip augmentation constants
# =========================================================

SELF_FLIP_H = [11, 12, 13, 15, 17, 18, 22, 26, 30, 35]

SELF_FLIP_V = [1, 5, 12, 15, 17]

SELF_FLIP_HV = [32, 40]

CROSS_FLIP = {
    19: 20,
    20: 19,
    33: 34,
    34: 33,
    36: 37,
    37: 36,
    38: 39,
    39: 38
}


# =========================================================
# Flip augmentation
# =========================================================

def flip_extend(X, y):
    """
    Extends dataset using symmetry-aware flipping.
    """

    X_out = []
    y_out = []

    for c in range(NUM_CLASSES):

        mask = (y == c)

        imgs = X[mask]

        # original images
        X_out.append(imgs)

        y_out.append(np.full(len(imgs), c))

        # self horizontal flips
        if c in SELF_FLIP_H:

            flipped = imgs[:, :, ::-1, :]

            X_out.append(flipped)

            y_out.append(np.full(len(flipped), c))

        # cross-class flips
        if c in CROSS_FLIP:

            src_class = CROSS_FLIP[c]

            flipped = X[y == src_class][:, :, ::-1, :]

            X_out.append(flipped)

            y_out.append(np.full(len(flipped), c))

    X_all = np.concatenate(X_out, axis=0)

    y_all = np.concatenate(y_out, axis=0)

    # vertical flips
    X_v = []
    y_v = []

    for c in SELF_FLIP_V:

        mask = (y_all == c)

        flipped = X_all[mask][:, ::-1, :, :]

        X_v.append(flipped)

        y_v.append(np.full(len(flipped), c))

    # horizontal + vertical flips
    X_hv = []
    y_hv = []

    for c in SELF_FLIP_HV:

        mask = (y_all == c)

        flipped = X_all[mask][:, ::-1, ::-1, :]

        X_hv.append(flipped)

        y_hv.append(np.full(len(flipped), c))

    X_final = np.concatenate(
        [X_all] + X_v + X_hv,
        axis=0
    )

    y_final = np.concatenate(
        [y_all] + y_v + y_hv,
        axis=0
    )

    return X_final, y_final.astype(np.int32)


# =========================================================
# Geometric augmentation
# =========================================================

def jitter_batch(X, intensity=0.75):

    return np.array([
        _jitter(img, intensity)
        for img in X
    ])


def _jitter(img, intensity):

    img = _rotate(img, intensity)

    img = _project(img, intensity)

    return img


def _rotate(img, intensity):

    delta = 30.0 * intensity

    angle = random.uniform(-delta, delta)

    return rotate(
        img,
        angle,
        mode='edge'
    )


def _project(img, intensity):

    size = img.shape[0]

    d = size * 0.3 * intensity

    def r():
        return random.uniform(-d, d)

    src = np.float32([
        [r(), r()],
        [r(), size - r()],
        [size - r(), size - r()],
        [size - r(), r()]
    ])

    dst = np.float32([
        [0, 0],
        [0, size],
        [size, size],
        [size, 0]
    ])

    t = ProjectiveTransform()

    # deprecated warning is harmless
    t.estimate(src, dst)

    return warp(
        img,
        t,
        output_shape=(size, size),
        order=1,
        mode='edge'
    )


# =========================================================
# Extended dataset generation
# =========================================================

def make_extended(X, y, multiplier=20, intensity=0.75):
    """
    Creates larger dataset using random jittering.
    """

    extras = []

    for _ in range(multiplier - 1):

        extras.append(
            jitter_batch(X, intensity)
        )

    X_ext = np.concatenate([X] + extras)

    y_ext = np.tile(y, multiplier)

    return X_ext, y_ext


# =========================================================
# Balanced dataset generation
# =========================================================

def make_balanced(X, y, per_class=PER_CLASS, intensity=0.75):
    """
    Creates balanced dataset with equal samples per class.
    """

    X_out = []

    y_out = []

    for c in range(NUM_CLASSES):

        imgs = X[y == c]

        print(f"Class {c}: {len(imgs)} samples")

        # IMPORTANT FIX
        if len(imgs) == 0:

            print(f"WARNING: skipping empty class {c}")

            continue

        needed = per_class - len(imgs)

        if needed > 0:

            idx = np.random.choice(
                len(imgs),
                needed,
                replace=True
            )

            extra = jitter_batch(
                imgs[idx],
                intensity
            )

            imgs = np.concatenate([
                imgs,
                extra
            ])

        X_out.append(
            imgs[:per_class]
        )

        y_out.extend(
            [c] * per_class
        )

    return (
        np.concatenate(X_out),
        np.array(y_out)
    )
