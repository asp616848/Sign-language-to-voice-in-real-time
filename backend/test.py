import cv2
import numpy as np
import mediapipe as mp
import requests

VIDEO_PATH = "backend/testy.mp4"
BACKEND_URL = "http://127.0.0.1:5000/predict"
DESIRED_FRAMES = 200
PAD_VALUE = 0.0

# Landmark indices (same as frontend)
FILTERED_INDICES = {
    "pose": [11, 12, 13, 14, 15, 16],
    "hands": list(range(21)),
    "face": [
        0, 4, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61,
        63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107,
        109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155,
        157, 158, 159, 160, 161, 162, 163, 172, 173, 176, 178, 181, 185, 191, 234,
        246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293,
        295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332,
        334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381,
        382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415,
        454, 466, 468, 473
    ]
}


mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)

def get_filtered_landmarks(results):
    frame_landmarks = []

    def extract_component(lms, indices, fallback_len):
        if lms:
            return [[lms.landmark[i].x, lms.landmark[i].y, lms.landmark[i].z] for i in indices]
        else:
            return [[PAD_VALUE, PAD_VALUE, PAD_VALUE]] * fallback_len

    frame_landmarks.extend(extract_component(results.pose_landmarks, FILTERED_INDICES["pose"], len(FILTERED_INDICES["pose"])))
    frame_landmarks.extend(extract_component(results.left_hand_landmarks, FILTERED_INDICES["hands"], len(FILTERED_INDICES["hands"])))
    frame_landmarks.extend(extract_component(results.right_hand_landmarks, FILTERED_INDICES["hands"], len(FILTERED_INDICES["hands"])))
    frame_landmarks.extend(extract_component(results.face_landmarks, FILTERED_INDICES["face"], len(FILTERED_INDICES["face"])))

    return frame_landmarks  # shape: (N, 3)

def extract_video_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened() and len(frames) < DESIRED_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        landmarks = get_filtered_landmarks(results)
        frames.append(landmarks)

    cap.release()

    # Pad if needed
    if len(frames) < DESIRED_FRAMES:
        empty_frame = [[PAD_VALUE, PAD_VALUE, PAD_VALUE]] * (
            len(FILTERED_INDICES["pose"]) +
            2 * len(FILTERED_INDICES["hands"]) +
            len(FILTERED_INDICES["face"])
        )
        frames += [empty_frame] * (DESIRED_FRAMES - len(frames))

    return np.array(frames[:DESIRED_FRAMES])

def send_to_backend(landmarks):
    try:
        response = requests.post(
            BACKEND_URL,
            json={"landmarks": landmarks.tolist()},
            headers={"Content-Type": "application/json"}
        )
        print("✅ Prediction:", response.json())
    except Exception as e:
        print("❌ Failed:", str(e))

if __name__ == "__main__":
    print("📽️  Processing video:", VIDEO_PATH)
    landmark_frames = extract_video_landmarks(VIDEO_PATH)
    print("📡 Sending to backend...")
    send_to_backend(landmark_frames)