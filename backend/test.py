import requests
import numpy as np

# Simulate a single frame: shape [543, 3]
sequence = np.random.rand(543, 3).astype(np.float32).tolist()

res = requests.post("http://127.0.0.1:5000/predict", json={"sequence": sequence})
print("Status:", res.status_code)
print("Prediction:", res.json())
