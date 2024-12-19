# Deep-learningProject

# Image Fault Detection Using CNN

## Overview
This project aims to detect faults in casting products using Convolutional Neural Networks (CNNs). The study involves training and evaluating multiple architectures to identify the most effective model for this task. The dataset is prepared, models are trained, and their performances are compared systematically.

---

## Architectures and Implementation

### 1. Pretrained CNN Architectures
#### Overview
Pretrained models such as Xception and DenseNet are employed using transfer learning. These architectures are initialized with weights trained on ImageNet to accelerate learning and improve accuracy for the fault detection task.

#### Steps
1. **Load Base Model**: Pretrained models are imported, and their top (classification) layers are removed.
2. **Add Custom Layers**: A new classification head is added with layers for feature extraction and prediction.
3. **Compile Model**: The model is compiled with binary cross-entropy loss and the Adam optimizer.
4. **Train and Validate**: Data augmentation is applied, and early stopping is used to prevent overfitting.

#### References
- Chollet et al., "Xception: Deep Learning with Depthwise Separable Convolutions," CVPR, 2017.  
- Huang et al., "Densely Connected Convolutional Networks," CVPR, 2017.

---

### 2. Custom ResNet Architecture
#### Overview
A custom ResNet50-like architecture is implemented to provide flexibility in handling the casting dataset.

#### Steps
1. **Define Bottleneck Block**: A building block consisting of convolutional layers with batch normalization and ReLU activation.
2. **Build the Network**: Assemble layers into a deep network with residual connections.
3. **Compile and Train**: Use binary cross-entropy loss and monitor validation loss to implement early stopping.

#### References
- He et al., "Deep Residual Learning for Image Recognition," CVPR, 2016.

---

## Results and Comparison

### Model Evaluation
Metrics: Precision, Recall, F1-score, Accuracy, and AUC score. Test set results are used to compare models.

### Classification Metrics

#### Xception Model
- **Defective**: Precision = 0.87, Recall = 0.89, F1-score = 0.88  
- **OK**: Precision = 0.81, Recall = 0.77, F1-score = 0.79  
- **Accuracy**: 85%  
- **Macro Average**: Precision = 0.84, Recall = 0.83, F1-score = 0.84  
- **Weighted Average**: Precision = 0.85, Recall = 0.85, F1-score = 0.85  

#### DenseNet Model
- **Defective**: Precision = 1.00, Recall = 0.90, F1-score = 0.95  
- **OK**: Precision = 0.85, Recall = 1.00, F1-score = 0.92  
- **Accuracy**: 94%  
- **Macro Average**: Precision = 0.93, Recall = 0.95, F1-score = 0.93  
- **Weighted Average**: Precision = 0.95, Recall = 0.94, F1-score = 0.94  

#### ResNet50 Model
- **Defective**: Precision = 0.94, Recall = 0.92, F1-score = 0.93  
- **OK**: Precision = 0.86, Recall = 0.90, F1-score = 0.88  
- **Accuracy**: 91%  
- **Macro Average**: Precision = 0.90, Recall = 0.91, F1-score = 0.91  
- **Weighted Average**: Precision = 0.91, Recall = 0.91, F1-score = 0.91  

---

### Comparison Table
| Metric       | Xception       | DenseNet       | ResNet50       |
|--------------|----------------|----------------|----------------|
| **Accuracy** | 85%            | 94%            | 91%            |
| **Precision**| 0.84 (macro avg)| 0.93 (macro avg)| 0.90 (macro avg)|
| **Recall**   | 0.83 (macro avg)| 0.95 (macro avg)| 0.91 (macro avg)|
| **F1-score** | 0.84 (macro avg)| 0.93 (macro avg)| 0.91 (macro avg)|
| **AUC**      | 0.9375         | 0.9965         | 0.9756         |

---

### Pros and Cons

#### Xception
- **Pros**: Lightweight, faster to train due to fewer parameters.  
- **Cons**: Lower accuracy compared to DenseNet and ResNet.

#### DenseNet
- **Pros**: Highest accuracy and F1-score. Benefits from densely connected layers which improve gradient flow.  
- **Cons**: Computationally intensive, requires more memory.

#### ResNet50
- **Pros**: Balanced performance with a good trade-off between accuracy and computational cost. Residual connections prevent vanishing gradients.  
- **Cons**: Accuracy slightly lower than DenseNet.

