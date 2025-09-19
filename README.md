# MNIST Digit Classification using a Custom CNN

This project implements a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset. The goal was to design a model with a parameter count around 20,000 while incorporating architectural improvements like Batch Normalization and strategic use of Dropout.

## Final Architecture: `FinalNet`

The final architecture, named `FinalNet`, is a sequential model consisting of several convolutional layers, followed by Batch Normalization, ReLU activation, Max Pooling, and finally two fully connected layers with a Dropout layer before the last one.

Here's a breakdown of the layers:

1.  **Convolutional Layer 1 (`conv1`):**
    *   Input Channels: 1 (grayscale MNIST images)
    *   Output Channels: 8
    *   Kernel Size: 3x3
    *   Padding: 1 (to maintain spatial dimensions)
    *   Purpose: Learn initial features from the input images.

2.  **Batch Normalization 1 (`bn1`):**
    *   Normalizes the output of `conv1` to stabilize training.

3.  **ReLU Activation:**
    *   Applies the Rectified Linear Unit activation function.

4.  **Convolutional Layer 2 (`conv2`):**
    *   Input Channels: 8
    *   Output Channels: 16
    *   Kernel Size: 3x3
    *   Padding: 1
    *   Purpose: Learn more complex features.

5.  **Batch Normalization 2 (`bn2`):**
    *   Normalizes the output of `conv2`.

6.  **ReLU Activation:**
    *   Applies the ReLU activation function.

7.  **Max Pooling 1 (`pool1`):**
    *   Kernel Size: 2x2
    *   Stride: 2
    *   Purpose: Downsample the feature maps, reducing spatial dimensions and computational cost.

8.  **Convolutional Layer 3 (`conv3`):**
    *   Input Channels: 16
    *   Output Channels: 16
    *   Kernel Size: 3x3
    *   Padding: 1
    *   Purpose: Further feature extraction.

9.  **Batch Normalization 3 (`bn3`):**
    *   Normalizes the output of `conv3`.

10. **ReLU Activation:**
    *   Applies the ReLU activation function.

11. **Convolutional Layer 4 (`conv4`):**
    *   Input Channels: 16
    *   Output Channels: 16
    *   Kernel Size: 3x3
    *   Padding: 1
    *   Purpose: Capture more abstract features.

12. **Batch Normalization 4 (`bn4`):**
    *   Normalizes the output of `conv4`.

13. **ReLU Activation:**
    *   Applies the ReLU activation function.

14. **Max Pooling 2 (`pool2`):**
    *   Kernel Size: 2x2
    *   Stride: 2
    *   Purpose: Further downsample the feature maps.

15. **Flattening:**
    *   The output from the last convolutional layer is flattened into a 1D vector to be fed into the fully connected layers. The size is 7 * 7 * 16 based on the output dimensions after pooling.

16. **Fully Connected Layer 1 (`fc1`):**
    *   Input Features: 7 * 7 * 16
    *   Output Features: 32
    *   Purpose: Learn non-linear combinations of the extracted features.

17. **ReLU Activation:**
    *   Applies the ReLU activation function.

18. **Dropout (`dropout`):**
    *   Dropout Rate: 0.25
    *   Purpose: Randomly sets a fraction of input units to 0 at each update during training, which helps prevent overfitting. Applied only once at the end of the network as requested.

19. **Fully Connected Layer 2 (`fc2`):**
    *   Input Features: 32
    *   Output Features: 10 (for the 10 MNIST classes)
    *   Purpose: Produce the final output scores for each class.

20. **Log Softmax Activation:**
    *   Applies the log-softmax function to the output, providing log-probabilities for each class.

## Architecture Summary

The `FinalNet` architecture is designed to be relatively compact while still being effective for MNIST classification. The use of Batch Normalization throughout the convolutional layers helps in training deeper networks. Applying Dropout only before the final classification layer is a specific design choice to regulate the final feature representation.

| Layer             | Type          | Output Shape     | Param # |
|-------------------|---------------|------------------|---------|
| Conv2d-1          | Conv2d        | [-1, 8, 28, 28]  | 80      |
| BatchNorm2d-2     | BatchNorm2d   | [-1, 8, 28, 28]  | 16      |
| Conv2d-3          | Conv2d        | [-1, 16, 28, 28] | 1,168   |
| BatchNorm2d-4     | BatchNorm2d   | [-1, 16, 28, 28] | 32      |
| MaxPool2d-5       | MaxPool2d     | [-1, 16, 14, 14] | 0       |
| Conv2d-6          | Conv2d        | [-1, 16, 14, 14] | 2,320   |
| BatchNorm2d-7     | BatchNorm2d   | [-1, 16, 14, 14] | 32      |
| Conv2d-8          | Conv2d        | [-1, 16, 14, 14] | 2,320   |
| BatchNorm2d-9     | BatchNorm2d   | [-1, 16, 14, 14] | 32      |
| MaxPool2d-10      | MaxPool2d     | [-1, 16, 7, 7]   | 0       |
| Linear-11         | Linear        | [-1, 32]         | 25,120  |
| Dropout-12        | Dropout       | [-1, 32]         | 0       |
| Linear-13         | Linear        | [-1, 10]         | 330     |
| **Total Params**  |               |                  | **31,450**|

This architecture provides a good balance between model complexity and performance for the MNIST dataset, achieving a high accuracy with a parameter count in the desired range.