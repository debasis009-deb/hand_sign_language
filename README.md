# 🤟 Hand Sign Language Recognition

A real-time American Sign Language (ASL) alphabet recognition web application powered by a custom Convolutional Neural Network (CNN) and MediaPipe.

## 🌟 Features

* **Real-time Recognition**: Translates hand signs from your webcam into text instantly.
* **Smart Preprocessing**: Uses Google's MediaPipe Tasks API to accurately detect and crop the hand region, eliminating background noise and improving prediction accuracy.
* **Custom CNN Architecture**: A deep Convolutional Neural Network trained on the **Sign MNIST** dataset, achieving over 99% accuracy on the test set.
* **Modern UI**: A beautiful, responsive, dark-themed glassmorphic user interface.
* **Word Builder**: String letters together to build words and sentences seamlessly.
* **Prediction History**: Keeps track of recent predictions with confidence scores.

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.8+ installed on your system.

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/hand_sign_language.git
cd hand_sign_language
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the App

The repository includes a pre-trained model (`sign_model.pth`). You can run the web app immediately:

```bash
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5000`. Allow camera permissions when prompted.

## 🧠 Training Your Own Model

If you want to train the model from scratch, you will need the Sign MNIST dataset:

1. Download the `sign_mnist_train.csv` and `sign_mnist_test.csv` datasets.
2. Place them in the root directory of this project.
3. Run the training script:
```bash
python train_model.py
```
This will train the CNN and generate a new `sign_model.pth` file.

## 🛠️ Built With

* **[Flask](https://flask.palletsprojects.com/)** - Backend web framework
* **[PyTorch](https://pytorch.org/)** - Deep learning framework for the CNN
* **[MediaPipe](https://developers.google.com/mediapipe)** - Hand detection and tracking (Tasks API)
* **[OpenCV](https://opencv.org/)** - Image processing
* **[Vanilla JS/CSS]** - Frontend logic and styling

## 📝 Notes

* The model supports 24 letters of the ASL alphabet (A-Y, excluding J and Z as they require motion).
* For best results, ensure your hand is clearly visible and well-lit.
