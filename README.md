# Traffic-sign-classification
Deep learning CNN-based traffic sign classification using image data.

# 🚦 Traffic Signal Classification Using Deep Learning (CNN)

## 1. Problem Statement

Traffic signal recognition is a critical component of **Advanced Driver Assistance Systems (ADAS)** and **autonomous driving**.  
The objective of this project is to build a **Convolutional Neural Network (CNN)** model that accurately classifies traffic signal images into their respective categories.

This is a **multi-class image classification problem** solved using deep learning techniques.

---

## 2. Dataset Overview

- **Dataset Type**: Image dataset of traffic signs
- **Problem Type**: Supervised Learning – Multi-class Classification
- **Input**: Traffic sign images
- **Target**: Traffic sign class label

### Dataset Characteristics
- Images resized to a uniform shape
- Multiple traffic sign categories
- Real-world variations in lighting and orientation

---

## 3. Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- OpenCV (for image processing)
- Scikit-learn

---

## 4. Data Preprocessing

- Image resizing and normalization
- Conversion of images to NumPy arrays
- Label encoding for class labels
- Train-test split for model validation
- Pixel value scaling for faster convergence

---

## 5. Model Architecture

A **Convolutional Neural Network (CNN)** was designed with:

- Convolutional layers for feature extraction
- ReLU activation functions
- MaxPooling layers for spatial reduction
- Fully connected dense layers
- Softmax activation for multi-class output

The architecture balances **model complexity and performance**.

---

## 6. Model Training

- Loss Function: Categorical Crossentropy
- Optimizer: Adam
- Evaluation Metric: Accuracy
- Trained for multiple epochs with validation monitoring

---

## 7. Model Evaluation

### Metrics Used
- Training Accuracy
- Validation Accuracy
- Loss curves

### Observations
- CNN successfully learned discriminative features
- Stable convergence observed during training
- Minimal overfitting due to proper preprocessing

---

## 8. Key Findings

- CNN outperforms traditional ML models for image data
- Image normalization significantly improves training stability
- Deep learning is effective for traffic signal recognition tasks
- Model demonstrates strong generalization capability

---

## 9. Conclusion

This project demonstrates an **end-to-end deep learning pipeline** for image classification, including data preprocessing, CNN model design, training, and evaluation.  
The trained model can serve as a foundation for real-world traffic sign recognition systems.

---

## 10. Repository Structure

Traffic-Signal-Classification/
│
├── Traffic_signal_AI_Classification.ipynb
├── README.md

---

## 11. Future Enhancements

- Data augmentation for better generalization
- Hyperparameter tuning of CNN architecture
- Transfer learning using pre-trained models (ResNet, MobileNet)
- Model deployment using TensorFlow Lite for edge devices

---

## 12. Author

Sathish V  
M.Tech – Signal Processing (NIT Calicut)  
Aspiring Data Scientist | Deep Learning

---
