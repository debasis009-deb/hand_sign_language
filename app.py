from flask import Flask, request, jsonify
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from model import load_model, LABEL_TO_LETTER
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = Flask(__name__, static_folder='static', template_folder='static')

# Load model once at startup
model = load_model()
model.eval()

# Check if model is trained
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sign_model.pth')
is_trained = os.path.exists(MODEL_PATH)

# Label index (0-23) to letter mapping
IDX_TO_LETTER = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P',
    15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W',
    22: 'X', 23: 'Y'
}

# Setup MediaPipe Hand Landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE)
hand_landmarker = vision.HandLandmarker.create_from_options(options)


def preprocess_frame(frame):
    """
    Preprocess a webcam frame to match Sign MNIST format.
    - Uses MediaPipe to precisely find the hand
    - Crops out just the hand with some padding
    - Converts to grayscale, resizes to 28x28, normalizes
    """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = hand_landmarker.detect(mp_image)

    if not results.hand_landmarks:
        return None

    h, w, _ = frame.shape
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    
    # Get bounding box for the hand
    for lm in results.hand_landmarks[0]:
        x, y = int(lm.x * w), int(lm.y * h)
        if x < x_min: x_min = x
        if x > x_max: x_max = x
        if y < y_min: y_min = y
        if y > y_max: y_max = y
        
    # Add padding to ensure the whole hand is captured
    pad = 30
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(w, x_max + pad)
    y_max = min(h, y_max + pad)
    
    # Crop the hand region
    hand_crop = frame[y_min:y_max, x_min:x_max]
    
    if hand_crop.size == 0:
        return None
        
    # Convert to grayscale
    gray_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
    
    # Resize to 28x28 (Sign MNIST size)
    resized = cv2.resize(gray_crop, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Reshape for model: (1, 1, 28, 28)
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    
    return tensor


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/status')
def status():
    return jsonify({'trained': is_trained})


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return jsonify({'error': 'Invalid image'}), 400

    try:
        input_tensor = preprocess_frame(frame)
        
        if input_tensor is None:
            return jsonify({'prediction': '', 'confidence': 0.0})

        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)
            confidence, class_idx = torch.max(probs, dim=1)
            confidence = confidence.item()
            class_idx = class_idx.item()
        
        letter = IDX_TO_LETTER.get(class_idx, '?')
        
        # Adjust confidence threshold slightly since background might still affect it a bit
        if confidence < 0.2:
            return jsonify({'prediction': '', 'confidence': 0.0})
        
        return jsonify({'prediction': letter, 'confidence': confidence})
    
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'prediction': '', 'confidence': 0.0, 'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)