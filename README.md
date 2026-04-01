# 🫁 LuminaScan AI: Minimalist Pneumonia Detection

LuminaScan AI is a modern, high-precision medical imaging desktop application built to assist in the detection of pneumonia from chest X-ray scans. By leveraging a custom-trained **YOLOv5m** (Medium) deep learning model and a sleek **PyQt5** minimalist interface, LuminaScan provides real-time diagnostic insights with confidence scoring.

---

## ✨ Key Features
*   **Minimalist Design:** A clean, "glass-style" medical dashboard that eliminates clutter.
*   **Deep Learning Core:** Powered by YOLOv5m for robust feature extraction and localization of lung consolidations.
*   **Confidence Scoring:** Provides a specific percentage (%) score for every detection to assist clinical judgment.
*   **Dynamic UI Feedback:** The interface adapts its color palette (Soft Green/Red) based on diagnostic findings.
*   **High Performance:** Optimized for fast inference using PyTorch Hub integration.

---

## 📸 Screenshots
<img width="1920" height="974" alt="image" src="https://github.com/user-attachments/assets/cc5fa709-92b0-4f54-b058-7590a7c5aec6" />


---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+
- A trained YOLOv5 model file (`best.pt`)

### 2. Installation
Clone the repository and install the dependencies:

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/LuminaScan-AI.git
cd LuminaScan-AI

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
