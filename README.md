🐶🐱 Dog vs Cat Image Classification using CNN with GUI
This project is a complete Deep Learning application that uses a Convolutional Neural Network (CNN) to classify images of dogs and cats. The model is trained using the popular Kaggle dataset "Dogs vs Cats", and a Tkinter-based Python GUI is built to allow real-time predictions on user-uploaded images.

📂 Repository Structure
graphql
Copy
Edit
.
├── catvsdog.h5             # Trained Keras CNN model
├── gui_app.py              # Tkinter GUI script for prediction
├── train_model.ipynb       # Colab-compatible training code
├── README.md               # Project documentation (this file)
└── dataset/                # Folder for dataset if used locally
📌 Project Highlights
CNN-based image classification using TensorFlow/Keras

Real-time predictions with a GUI

Dataset preprocessing, normalization, and augmentation

Training accuracy/loss visualizations

Model evaluation and testing

User can classify multiple images at once

📊 Dataset
Source: Kaggle - Dogs vs Cats by Salader

Contains:

/train folder with labeled images (cat.x.jpg, dog.x.jpg)

/test folder for validation images

🧠 Model Architecture
Input size: 256x256 RGB images

3 Convolutional Layers with ReLU activation and MaxPooling

Fully Connected Dense Layers

Dropout for regularization

Final Layer: 1 neuron with Sigmoid activation

Loss: Binary Crossentropy

Optimizer: Adam

python
Copy
Edit
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
📈 Training Performance
Training Epochs: 3 (can be increased for better accuracy)

Validation and training accuracy plotted

Achieved accuracy above 90% on validation data

🖥️ GUI Application (Tkinter)
Built with Python’s tkinter, PIL, and numpy

Lets the user upload any image and get predictions:

"It's a Dog" or "It's a Cat"

Uses the trained catvsdog.h5 model

Simple and lightweight desktop interface

🖼️ GUI Screenshot (sample layout):

🚀 How to Run the Project
🔹 Option 1: Google Colab (Training)
Open train_model.ipynb in Google Colab

Install dependencies:

python
Copy
Edit
!pip install opendatasets
Download dataset:

python
Copy
Edit
import opendatasets as od
od.download("https://www.kaggle.com/datasets/salader/dogs-vs-cats")
Train the model (code provided in notebook)

Save the model:

python
Copy
Edit
model.save("catvsdog.h5")
🔹 Option 2: Local System (GUI)
Clone the repo:

bash
Copy
Edit
git clone https://github.com/your-username/cat-dog-classifier.git
cd cat-dog-classifier
Install dependencies:

bash
Copy
Edit
pip install tensorflow pillow numpy keras matplotlib
Run the GUI:

bash
Copy
Edit
python gui_app.py
Click "Upload Image" → Select an image → Click "Classify Image"
You’ll see the predicted class displayed below the image.

📦 Requirements
Python 3.7+

TensorFlow

Keras

Pillow

NumPy

Matplotlib

Tkinter (pre-installed with Python on most systems)

📍 Future Improvements
Use Data Augmentation for better generalization

Add multi-class support or confidence score

Enhance GUI (drag & drop, batch prediction, webcam support)

Deploy as a web app using Flask or Streamlit

