# Multimodal Lung Cancer Prediction using CNN + ML (Streamlit App)

This project implements a **multimodal lung cancer prediction system** combining:
- A **tabular ML model** (Logistic Regression + StandardScaler)
- A **CNN model** trained on lung CT images

The final prediction is obtained using a weighted fusion of both model outputs.
A **Streamlit web application** provides a simple and interactive interface for users to upload data and view predictions.

---

## 🌐 Live Demo

[View the Live Application](https://lung-cancer-detection-myetsbam3kqsjcytg4pwev.streamlit.app/)

---

## 📁 Project Structure
```
project/
│
├── app.py                         # Streamlit UI
├── models/
│   ├── tabular_model.pkl          # ML model (loaded via joblib)
│   └── cnn_model.h5 (hosted externally)
├── requirements.txt
└── README.md
```

---

## 🤝 Model Hosting
The **CNN model (~58MB)** is hosted on **Hugging Face Hub**.

In `app.py`, the model is loaded as:
```python
from huggingface_hub import hf_hub_download
import tensorflow as tf

model_path = hf_hub_download(
    repo_id="SarthakBaghel1/cnn_model",
    filename="cnn_model.h5"
)
cnn_model = tf.keras.models.load_model(model_path)
```

This avoids pushing large model files to GitHub.

---

## 🚀 Running Locally

### 1. Clone the Repository

```bash
git clone https://github.com/SarthakBaghel/Lung-Cancer-Detection.git
cd Lung-Cancer-Detection
```

### 2. Create & activate virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the Streamlit App
```bash
streamlit run app.py
```

---

## 🌍 Deployment (Streamlit Cloud Recommended)

1. Push code to GitHub (models not required if hosted on HF)
2. Visit: https://streamlit.io/cloud
3. Connect your GitHub repository
4. Deploy

### Important Note:
Your `requirements.txt` must be minimal and Linux-compatible.  
Example used:

```
streamlit==1.48.1
numpy==1.26.4
pandas==2.3.1
joblib==1.5.1
scikit-learn==1.5.1
tensorflow==2.16.2
huggingface_hub==1.1.2
h5py==3.14.0
Pillow==11.3.0
requests==2.32.4
```

---

## 🧠 Model Fusion Logic

The final prediction combines both models with weighted averaging:

```python
tab_pred = tabular_model.predict_proba(data)[0][1]
cnn_pred = cnn_model.predict(image_preprocessed)[0][0]

final_score = (0.6 * cnn_pred) + (0.4 * tab_pred)
```

- **CNN weight (60%)**: Primary prediction from imaging analysis
- **Tabular weight (40%)**: Secondary prediction from clinical/tabular features

---

## 📌 Key Features
- Multimodal prediction approach
- Clean and interactive UI (Streamlit)
- Cloud-ready deployment configuration
- External model hosting for large models (Hugging Face)

---

## 🎯 Future Improvements
- Add explainability (Grad-CAM for CNN & SHAP for tabular model)
- Deploy GPU-backed API for faster inference
- Containerize with Docker

---

## 📝 License
This project is for educational and research purposes only.
