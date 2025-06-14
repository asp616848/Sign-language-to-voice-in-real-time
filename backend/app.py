from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Load your Keras model
MODEL_PATH = "results(1)/Kamel_Models/12-14.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Prediction route

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        raw_landmarks = data["landmarks"]  # Should be [200][<=180][3]

        # Fix: Ensure each frame has 180 landmarks, pad with [-100, -100, -100]
        fixed_landmarks = []
        for frame in raw_landmarks:
            if len(frame) < 180:
                pad_len = 180 - len(frame)
                frame += [[-100, -100, -100]] * pad_len
            elif len(frame) > 180:
                frame = frame[:180]
            fixed_landmarks.append(frame)

        if len(fixed_landmarks) < 200:
            fixed_landmarks += [[[-100, -100, -100]] * 180] * (200 - len(fixed_landmarks))
        elif len(fixed_landmarks) > 200:
            fixed_landmarks = fixed_landmarks[:200]

        # Now safe to convert to numpy array
        landmarks_np = np.array(fixed_landmarks).reshape(1, 200, 180, 3)

        # Predict
        prediction = model.predict(landmarks_np)
        predicted_class = int(np.argmax(prediction, axis=-1)[0])
        print({"\n\npredicted class is":predicted_class})
        return jsonify({"prediction": predicted_class})

    except Exception as e:
        print("🔥 Exception in /predict route:")
        print(e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
