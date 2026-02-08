🌿 Plant Leaf Disease Detection using CNN


📌 Project Overview

This project is a deep learning-based plant leaf disease detection system that classifies plant leaf images into different disease categories using a Convolutional Neural Network (CNN).

The goal of this project is to automatically detect plant diseases from leaf images and assist in early diagnosis, which can help farmers and researchers improve crop health and productivity.

The trained CNN model is integrated into a user-friendly application for real-time image-based predictions.


📊 Dataset

The model is trained using the PlantVillage dataset, which contains labeled images of healthy and diseased plant leaves.

🔗 Dataset link:
https://www.kaggle.com/datasets/emmarex/plantdisease

Dataset features:

Multiple plant species (Tomato, Potato, Pepper, etc.)

Healthy and diseased leaf classes

38 classification categories

Thousands of labeled images

Data preprocessing:

Image resizing to 128 × 128 pixels

RGB color conversion

Pixel normalization (scaling to 0–1)

Dataset splitting into training and validation sets


🧠 Model Architecture

A custom Convolutional Neural Network (CNN) was built using TensorFlow/Keras.

CNN structure:

Conv2D (32 filters) + MaxPooling

Conv2D (64 filters) + MaxPooling

Conv2D (128 filters) + MaxPooling

Flatten layer

Dense layer (128 neurons)

Dropout (0.5) for regularization

Output layer (38 classes, Softmax activation)

Training details:

Optimizer: Adam

Loss function: Sparse Categorical Crossentropy

Evaluation metric: Accuracy


🌐 Application

The trained model is integrated into an application that allows users to upload plant leaf images and receive disease predictions.

App features:

Image upload interface

Automatic image preprocessing

Real-time disease prediction

Clean and simple UI


🛠️ Technologies Used

Python

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

Streamlit (for deployment)

Jupyter Notebook

🎯 Project Outcomes

This project demonstrates:

Image preprocessing using OpenCV

Building a CNN for image classification

Deep learning model training and evaluation

Multi-class disease detection

Deployment of a deep learning model as an interactive app


🔮 Future Improvements

Improve model accuracy with transfer learning

Add more plant species and disease classes

Deploy as a mobile or cloud-based application

Add real-time camera detection


