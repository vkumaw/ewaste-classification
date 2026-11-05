# Automated E-Waste Classification Using Lightweight MobileNetV2 with Webcam Inference

An end-to-end computer vision project that classifies electronic waste across 10 distinct operational categories using a compact MobileNetV2 architecture with dynamic-range INT8 quantization and local webcam verification.

---

## 🔗 Live Kaggle Notebook (Executed Cells & Visuals)
> **Direct Link to Full Interactive Notebook**:  
> 👉 **[View Executed Kaggle Notebook with Complete Outputs](https://www.kaggle.com/code/vaidehikumawat/ewaste-project)**  
*(Contains full exploratory data analysis, training curves across 8 epochs, confusion matrix, test batch predictions, and model size benchmarks).*

---

## 📁 Repository Contents

| File Name | Description |
| :--- | :--- |
| `ewaste-project.ipynb` | Complete training, evaluation, confusion matrix generation, and TFLite quantization pipeline. |
| `run_demo.py` | Standalone Python script for real-time video verification via webcam with a region-of-interest (ROI) targeting box. |
| `model_quantized_int8.tflite` | 8-bit dynamic-range quantized model binary (~2.61 MB) optimized for low-resource inference. |
| `model_float32.tflite` | Unquantized baseline TFLite flatbuffer binary (~9.09 MB). |

---

## 📊 Model & Benchmark Summary

- **Architecture**: MobileNetV2 backbone (frozen ImageNet weights) + GlobalAveragePooling2D + Dense(128, ReLU) + Softmax(10).
- **Dataset Partitioning**: 10 balanced classes (Battery, Keyboard, Microwave, Mobile, Mouse, PCB, Player, Printer, Television, Washing Machine) totaling 3,000 images (2,400 train / 300 val / 300 test).
- **Test Set Accuracy**: 94.33% (283 / 300 correct classifications; Macro F1-score: 0.94).
- **Compression**: INT8 Quantization reduces storage footprint from 9,306.95 KB down to 2,615.91 KB (3.56x reduction).
- **Webcam Latency**: Single-frame inference executes in ~18–24 ms on standard CPU hardware.

---

## 💻 How to Run the Webcam Inference Locally

### 1. Requirements
Ensure Python 3.9+ is installed on your machine. Install the required runtime libraries:
```bash
pip install opencv-python pillow numpy tensorflow
```

### 2. Execution
Navigate to your project directory and launch the webcam script:

```bash
python run_demo.py
```

*(The script will automatically detect and load `model_float32.tflite` or `model_quantized_int8.tflite` from the same folder).*

### 3. Usage Guidelines
- **Target Inspection Area**: The webcam interface renders a centered 280 × 280 pixel green target box on the video stream to isolate the item from room background noise.
- **Positioning**: Hold an electronic item (keyboard, mouse, mobile phone, battery, etc.) directly inside the target box.
- **Testing via Phone Screen**: If showing photos displayed on a smartphone screen, reduce phone screen brightness to ~40–50% and tilt the device slightly to eliminate camera glare and reflections.
- **Real-Time Readout**: The top banner displays the predicted class, confidence percentage, and inference latency per frame.
- **Exit**: Press **`q`** on your keyboard at any time to close the camera window and release memory safely.

