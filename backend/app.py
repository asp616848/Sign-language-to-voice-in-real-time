from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import math

app = Flask(__name__)
CORS(app)

# Load your Keras model
MODEL_PATH = "results(1)/Kamel_Models/12-14.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# ===================== Preprocessing Utilities =====================

def skipping(landmarks, desired_frames, mode='floor'):
    frames_num = landmarks.shape[0]
    if frames_num == 0:
        return np.zeros((desired_frames, 180, 3))
    
    skip_factor = math.floor(frames_num / desired_frames) if mode == 'floor' else math.ceil(frames_num / desired_frames)
    skipped_landmarks = []

    for i in range(0, frames_num, skip_factor):
        skipped_landmarks.append(landmarks[i])
        if len(skipped_landmarks) == desired_frames:
            break

    while len(skipped_landmarks) < desired_frames:
        skipped_landmarks.append(np.full((180, 3), -100))

    return np.array(skipped_landmarks)

def cloning(landmarks, desired_frames):
    frames_num = landmarks.shape[0]
    if frames_num == 0:
        return np.full((desired_frames, 180, 3), -100)

    repeat_factor = math.ceil(desired_frames / frames_num)
    cloned_list = np.repeat(landmarks, repeat_factor, axis=0)
    return cloned_list[:desired_frames]

def clone_skip(landmarks, desired_frames=200):
    frames_number = landmarks.shape[0]

    if frames_number == desired_frames:
        return landmarks
    elif frames_number < desired_frames:
        return cloning(landmarks, desired_frames)
    else:
        return skipping(landmarks, desired_frames)

def preprocess_landmarks(raw_landmarks, desired_frames=200, desired_landmarks=180):
    fixed_frames = []
    for frame in raw_landmarks:
        if len(frame) < desired_landmarks:
            pad_len = desired_landmarks - len(frame)
            frame += [[-100, -100, -100]] * pad_len
        elif len(frame) > desired_landmarks:
            frame = frame[:desired_landmarks]
        fixed_frames.append(frame)

    landmarks_np = np.array(fixed_frames)
    landmarks_np = clone_skip(landmarks_np, desired_frames)
    return landmarks_np.reshape(1, desired_frames, desired_landmarks, 3)

# ===================== Prediction Route =====================

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        data = request.get_json()
        
        raw_landmarks = data["landmarks"]  # list of frames

        # Use your preprocessing function
        landmarks_np = preprocess_landmarks(raw_landmarks)
        print("Input shape:", landmarks_np.shape)
        print("Sample input:", landmarks_np[0, :20, :20, :])  # Print first 5 frames, 5 landmarks

        prediction = model.predict(landmarks_np)
        predicted_class = int(np.argmax(prediction, axis=-1)[0])
        print("🔮 Prediction logits:", predicted_class)
        return jsonify({"prediction": predicted_class})

    except Exception as e:
        print("🔥 Exception in /predict route:")
        print(e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)