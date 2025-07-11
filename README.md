# 🐶🐱 Dog vs Cat Image Classification using CNN with GUI

This is a Deep Learning project that classifies images of **cats** and **dogs** using a **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras**.  
It includes:
- A training notebook to build and save the model.
- A GUI application using **Tkinter** to perform real-time classification on any image.

---

## 📂 Project Structure

.
├── catvsdog.h5 # Trained Keras CNN model
├── gui_app.py # GUI application script
├── train_model.ipynb # Training code (Colab compatible)
├── dataset/ # Dataset folder (optional for local use)
└── README.md # Project documentation


---

## 🚀 Features

- ✅ CNN-based binary classification (Cat or Dog)
- 🖼️ Upload and classify any image via GUI
- 📊 Training and validation accuracy/loss visualization
- 📁 Dataset loaded from Kaggle using `opendatasets`
- 💾 Model saved in `.h5` format for reuse

## 📊 Model Details

- Input Size: 128x128 RGB images
- Layers: Conv2D, MaxPooling, Flatten, Dense
- Activation: ReLU and Sigmoid
- Loss Function: Binary Crossentropy
- Optimizer: Adam
- Accuracy: Achieved over 90% accuracy on validation data


---

## 📊 Dataset

- **Source**: [Kaggle – Dogs vs Cats](https://www.kaggle.com/datasets/salader/dogs-vs-cats)

---
## 🖼️ GUI Demo

- You can load your own image and the GUI will predict whether it's a cat or a dog using the trained model.
- --

## 🚀 How to Run

- 1 Clone the repository:
  
git clone https://github.com/<your-username>/cat-dog-classification.git

- 2 Install required libraries:

  pip install tensorflow pillow numpy keras matplotlib

- 3 Run the GUI:

python gui_app.py


## 📌 Requirements

Python 3.7+

TensorFlow / Keras

Pillow

NumPy

Matplotlib (for visualization)

Tkinter (for GUI)



  
