# Preprocessing check

`preprocess.js` must match the Python pipeline exactly, or predictions drift.

```bash
pip install opencv-python-headless==4.10.0.84 scikit-image==0.24.0 numpy==1.26.4
python reference.py      # writes cases.json from OpenCV + scikit-image
node compare.mjs         # both max diffs should be 0
```
