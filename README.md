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

2. Execution
Run the webcam detection script from your project directory:

Bash
python run_demo.py

3. Usage Guidelines
The script opens your webcam and renders a green Center Inspection Box .Hold an electronic item (or an image displayed on a phone screen) inside the green box.Tip for phone displays: Reduce screen brightness to ~40–50% to prevent camera glare from washing out component details.Press q at any time to close the stream and free system memory.

