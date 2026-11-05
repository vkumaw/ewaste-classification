from collections import deque
import os
import time
import cv2
import numpy as np

try:
  import tflite_runtime.interpreter as tflite
except ImportError:
  import tensorflow.lite as tflite

# 1. Exact 10 E-Waste Classes in training order
CLASS_NAMES = [
    'Battery',
    'Keyboard',
    'Microwave',
    'Mobile',
    'Mouse',
    'PCB',
    'Player',
    'Printer',
    'Television',
    'Washing Machine',
]

# 2. Load Model (Prefers Float32, falls back to INT8)
MODEL_PATH = 'model_float32.tflite'
if not os.path.exists(MODEL_PATH):
  MODEL_PATH = 'model_quantized_int8.tflite'

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f'Active Edge Model: {MODEL_PATH}')
print(f"Input Type: {input_details[0]['dtype']} (Raw [0-255] Input)")

# 3. Camera Setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

pred_history = deque(maxlen=5)

print('\nCamera stream active. Press "q" to exit.')

while True:
  ret, frame = cap.read()
  if not ret:
    break

  h, w, _ = frame.shape
  box_size = 280
  x1 = (w - box_size) // 2
  y1 = (h - box_size) // 2
  x2 = x1 + box_size
  y2 = y1 + box_size

  # Crop only the center targeting square
  roi = frame[y1:y2, x1:x2]
  rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

  # Resize to 160x160 matching model layer
  resized = cv2.resize(rgb_roi, (160, 160), interpolation=cv2.INTER_AREA)

  # Send raw [0, 255] floats directly (model pre-processes internally)
  input_tensor = np.expand_dims(resized.astype(np.float32), axis=0)

  # Run TFLite inference
  t0 = time.perf_counter()
  interpreter.set_tensor(input_details[0]['index'], input_tensor)
  interpreter.invoke()
  raw_preds = interpreter.get_tensor(output_details[0]['index'])[0].astype(
      np.float32
  )
  latency_ms = (time.perf_counter() - t0) * 1000.0

  # Softmax verification
  if np.max(raw_preds) > 1.0 or np.min(raw_preds) < 0.0:
    exp_preds = np.exp(raw_preds - np.max(raw_preds))
    preds = exp_preds / np.sum(exp_preds)
  else:
    preds = raw_preds

  # Smooth across last 5 frames
  pred_history.append(preds)
  avg_preds = np.mean(pred_history, axis=0)

  top_indices = np.argsort(avg_preds)[::-1][:3]
  best_class = CLASS_NAMES[top_indices[0]]
  confidence = avg_preds[top_indices[0]] * 100.0

  # UI Overlay
  tag_color = (0, 255, 0) if confidence > 50 else (0, 165, 255)
  cv2.rectangle(frame, (x1, y1), (x2, y2), tag_color, 2)
  cv2.putText(
      frame,
      f'{best_class}: {confidence:.1f}%',
      (20, 42),
      cv2.FONT_HERSHEY_SIMPLEX,
      1.0,
      tag_color,
      2,
      cv2.LINE_AA,
  )
  cv2.putText(
      frame,
      f'CPU Latency: {latency_ms:.1f}ms',
      (20, 75),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.55,
      (200, 200, 200),
      1,
      cv2.LINE_AA,
  )

  cv2.imshow('Edge E-Waste Sorter Demo', frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()