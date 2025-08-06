# 🧠 Brain Tumor Detection & Segmentation

A comprehensive deep learning solution for brain tumor **detection** and **segmentation** using **Keras-based ResNet** and **U-Net** architectures, deployed through an interactive **Flask web interface**.

---

## 🎯 Project Overview

This project delivers an end-to-end pipeline that performs:
- **Classification**: Tumor vs Healthy using ResNet-18 (TensorFlow/Keras)
- **Segmentation**: Pixel-wise tumor detection using U-Net

Achieves:
- **92.16% classification accuracy**
- **99.63% segmentation accuracy**

---

## 🔬 Key Features

- 🧠 **Dual Models**: 
  - ResNet for binary tumor classification
  - U-Net for segmentation mask prediction
- ⚙️ **Deep Learning Stack**:
  - Keras (TensorFlow backend)
  - No PyTorch dependencies
- 🖼️ **Real-time Image Analysis** via Flask
- 📈 **Accurate Results** for both detection and localization

---

## 📊 Dataset Overview

### Classification Dataset
- **Source**: Brain MRI Images for Brain Tumor Detection (Kaggle)
- **Classes**: Tumor / No Tumor
- **Split**: 80% Training, 20% Testing

### Segmentation Dataset
- **Source**: LGG MRI Segmentation Dataset
- **Labels**: Pixel-wise binary tumor masks
- **Format**: TIFF images + masks
- **Split**: 70% Training, 15% Validation, 15% Testing

---

## 🧱 Project Structure
```
brain_tumor_project/
│
├── best_model.keras
│
├── brain_tumor_segmentation_model.keras
│
├── deployment/
│ ├── app.py
│ ├── utils/
│ │ ├── predictor.py
│ ├── static/
│ │ ├── uploads/
│ │ ├── results/
│ │ └── style.css
│ └── templates/
│ └── index.html
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🧪 Model Performance

### ✅ ResNet Classification

- **Test Accuracy**: 92.16%
- **Architecture**: ResNet (Transfer Learning)
- **Training**: 40 epochs with early stopping
- **Loss**: Categorical cross-entropy

📌 Example from final training:
Train Accuracy: 94.47%
Validation Accuracy: 90.47%
Train Loss: 0.22
Val Loss: 0.25


---

### 🧠 U-Net Segmentation Performance
- **Training Accuracy**: 99.72%
- **Validation Accuracy**: 99.66%
- **Dice Coefficient (Val)**: **0.8263**
- **IoU Score (Val)**: ~0.70
- **Loss Function**: Negative Dice Loss

### 📜 License
This project is licensed under the MIT License – see the LICENSE file for details.


