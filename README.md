# 🌾 PRCP-1001 – Rice Leaf Disease Detection

## 📌 Project Overview
Rice crops are highly susceptible to diseases that can significantly reduce yield and quality. Early identification of leaf diseases helps farmers take preventive measures.

This project implements a **Deep Learning-based image classification system** to detect major rice leaf diseases using:

- Convolutional Neural Networks (CNN)
- Transfer Learning (MobileNetV2)
- Data Augmentation
- Streamlit Frontend

---

## 🎯 Objectives
✔ Perform Exploratory Data Analysis (EDA)  
✔ Build a CNN model from scratch  
✔ Apply Transfer Learning using MobileNetV2  
✔ Analyze Data Augmentation impact  
✔ Compare model performances  
✔ Develop Streamlit-based UI  

---

## 📂 Dataset Description
The dataset contains **120 JPG images** of rice leaves categorized into three disease classes:

| Disease Class | Number of Images |
|--------------|------------------|
| Leaf Smut | 39 |
| Brown Spot | 40 |
| Bacterial Leaf Blight | 40 |

The dataset is **nearly balanced** with minor class imbalance.

---

## 🧪 Technologies Used
- Python
- NumPy, Pandas
- Matplotlib, Seaborn
- OpenCV
- Scikit-learn
- TensorFlow / Keras
- Streamlit

---

## 🧠 Models Implemented

### ✅ 1. Custom CNN
- Conv2D Layers
- MaxPooling
- Dense Layers

### ✅ 2. Transfer Learning (MobileNetV2)
- Pretrained ImageNet weights
- Frozen base layers
- Custom classification head

---

## 🎭 Data Augmentation
Applied techniques:

✔ Rotation  
✔ Zooming  
✔ Horizontal Flip  

Purpose:

- Increase dataset diversity
- Reduce overfitting
- Improve generalization

---

## 📊 Model Evaluation
Performance measured using:

✔ Accuracy  
✔ Precision  
✔ Recall  
✔ F1-score  
✔ Confusion Matrix  

---

## 🏆 Best Model
✅ **MobileNetV2 (Transfer Learning)**

Reasons:

✔ Better generalization  
✔ Faster convergence  
✔ Suitable for small datasets  
✔ Lightweight architecture  

---

## ⚠️ Challenges Faced

| Challenge | Solution |
|----------|-----------|
| Small dataset | Data Augmentation |
| Overfitting risk | Dropout + Transfer Learning |
| Class similarity | Deep feature extraction |
| Minor imbalance | Stratified splitting |

---

## 💻 Streamlit Frontend
A user-friendly web interface was developed to:

✔ Upload rice leaf image  
✔ Predict disease  
✔ Show confidence score  
✔ Display class probabilities  

Run locally:

```bash
streamlit run streamlit_app.py
