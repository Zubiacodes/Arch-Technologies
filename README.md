Task 3: Real-Time Object Detection (YOLOv8)

We implemented real-time object detection using the YOLOv8 model from Ultralytics
.

🔹 How It Works

Load a pre-trained YOLOv8 model (yolov8n.pt).

Capture webcam feed using OpenCV.

Run YOLO detection frame by frame.

Display bounding boxes and class labels in real time.

Task 4: Facial Emotion Recognition (FER-2013 Dataset)

We built a Facial Emotion Recognition (FER) system using Convolutional Neural Networks (CNNs) trained on the FER-2013 dataset.

🔹 Dataset

FER-2013 dataset consists of 48x48 grayscale images.

It includes 7 emotion classes:

😀 Happy

😢 Sad

😡 Angry

😱 Fear

😲 Surprise

😐 Neutral

🤢 Disgust

🔹 Approach

Preprocess dataset (resize, normalize, grayscale).

Build and train a CNN using TensorFlow/Keras.

Evaluate accuracy on the validation set.

Test real-time predictions using webcam input.
