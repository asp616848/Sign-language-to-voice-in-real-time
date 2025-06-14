import requests
import json

url = "http://127.0.0.1:5000/predict"

# Generate dummy sequence data [543, 3]
sequence = [[0.0, 0.0, 0.0] for _ in range(543)]

response = requests.post(url, json={"sequence": sequence})

print("Status code:", response.status_code)
print("Response:", response.json())
