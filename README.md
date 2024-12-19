# Deep-learningProject

# Image Fault Detection Using CNN
##  Overview
 This project aims to detect faults in casting products using Convolutional Neural Networks (CNNs). The
 study involves training and evaluating multiple architectures to identify the most effective model for this
 task. The dataset is prepared, models are trained, and their performances are compared systematically.
 Architectures and Implementation
 1. Pretrained CNN Architectures
##  Overview
 Pretrained models such as Xception and DenseNet are employed using transfer learning. These
 architectures are initialized with weights trained on ImageNet to accelerate learning and improve
 accuracy for the fault detection task.
 b. Steps:
 1. Load Base Model: Pretrained models are imported and their top (classification) layers are removed.
 2. Add Custom Layers: A new classification head is added with layers for feature extraction and
 prediction.
 3. Compile Model: The model is compiled with a binary cross-entropy loss and Adam optimizer.
 4. Train and Validate: Data augmentation is applied, and early stopping is used to prevent overfitting.
 c. Graph
 d. Reference
 Chollet et al., "Xception: Deep Learning with Depthwise Separable Convolutions," CVPR, 2017.
 Huang et al., "Densely Connected Convolutional Networks," CVPR, 2017.
 2. Custom ResNet Architecture
 a. Overview

## Image Fault Detection Using CNN
 A custom ResNet50-like architecture is implemented to provide flexibility in handling the casting dataset.
 b. Steps:
 1. Define Bottleneck Block: A building block consisting of convolutional layers with batch
 normalization and ReLU activation.
 2. Build the Network: Assemble layers into a deep network with residual connections.
 3. Compile and Train: Use binary cross-entropy loss and monitor validation loss to implement early
 stopping.
 c. Graph
 d. Reference
 He et al., "Deep Residual Learning for Image Recognition," CVPR, 2016.
 Results and Comparison
 ### Model Evaluation
 Metrics: Precision, Recall, F1-score, Accuracy, and AUC score.
 Data: Test set results are used to compare models.
 Classification Metrics
 Xception Model
 Defective: Precision = 0.87, Recall = 0.89, F1-score = 0.88
 OK: Precision = 0.81, Recall = 0.77, F1-score = 0.79
 Accuracy: 85%
 Macro Average: Precision = 0.84, Recall = 0.83, F1-score = 0.84
 Weighted Average: Precision = 0.85, Recall = 0.85, F1-score = 0.85
 DenseNet Model
 Defective: Precision = 1.00, Recall = 0.90, F1-score = 0.95
 OK: Precision = 0.85, Recall = 1.00, F1-score = 0.92
 Accuracy: 94%
 Macro Average: Precision = 0.93, Recall = 0.95, F1-score = 0.93
 Weighted Average: Precision = 0.95, Recall = 0.94, F1-score = 0.94
 Image Fault Detection Using CNN
 ResNet50 Model
 Defective: Precision = 0.94, Recall = 0.92, F1-score = 0.93
 OK: Precision = 0.86, Recall = 0.90, F1-score = 0.88
 Accuracy: 91%
 Macro Average: Precision = 0.90, Recall = 0.91, F1-score = 0.91
 Weighted Average: Precision = 0.91, Recall = 0.91, F1-score = 0.91
 Comparison Table
 Metric
 Accuracy
 Precision
 Recall
 F1-score
 AUC
 Xception
 85%
 0.84 (macro avg)
 0.83 (macro avg)
 DenseNet
 94%
 ResNet50
 91%
 0.93 (macro avg)
 0.95 (macro avg)
 0.84 (macro avg)
 0.9375
 Pros and Cons
 a. Xception
 0.93 (macro avg)
 0.9965
 0.90 (macro avg)
 0.91 (macro avg)
 0.91 (macro avg)
 0.9756
