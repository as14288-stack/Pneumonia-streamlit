# Backend of the app: model loading, image pre-processing and prediction (no web code in here, so it can be tested on its own).
import io, json, os
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))

def load_config(path=None):
    return json.load(open(path or os.path.join(HERE, 'model_config.json')))

def load_model(path=None):
    import tensorflow as tf
    return tf.keras.models.load_model(path or os.path.join(HERE, 'pneumonia_model.keras'), compile=False)

def _to_uint8(a):
    a = a.astype(np.float32)
    return ((a - a.min()) / (a.max() - a.min() + 1e-8) * 255.0).astype(np.uint8)       # same window scaling as in training

def read_image(data, filename=''):
    """Return a 2-D uint8 chest X-ray from raw bytes of a DICOM (.dcm), PNG or JPG file."""
    if filename.lower().endswith('.dcm') or data[128:132] == b'DICM':
        import pydicom
        arr = pydicom.dcmread(io.BytesIO(data)).pixel_array
        if arr.ndim == 3: arr = arr[..., 0]
        return _to_uint8(arr)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None: raise ValueError('Could not read this file as an image.')
    return _to_uint8(img)

def preprocess(img_u8, size):
    """uint8 grayscale -> (1, size, size, 1) float32 in [0, 1]  (RGB->gray is a no-op for an already single-channel image)."""
    rgb = np.stack([img_u8] * 3, axis=-1)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return (gray.astype(np.float32) / 255.0)[None, ..., None]

def predict(model, img_u8, size, threshold):
    p = float(model.predict(preprocess(img_u8, size), verbose=0).ravel()[0])
    return dict(probability=p, label='Pneumonia likely' if p >= threshold else 'No pneumonia flagged', threshold=threshold)
