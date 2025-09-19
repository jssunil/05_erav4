# MNIST Digit Classification using a Custom CNN

This project implements a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset. The goal was to design a model with a parameter count around 20,000 while incorporating architectural improvements like Batch Normalization and strategic use of Dropout.

## Architectures Explored

We explored a few architectures to achieve the target parameter count of around 20,000.

### Architecture 1: `FinalNet`

The first architecture, named `FinalNet`, was designed with several convolutional and linear layers, incorporating Batch Normalization and a single Dropout layer at the end.

Here's a breakdown of the layers:

1.  **Convolutional Layer 1 (`conv1`):** Input: 1 channel, Output: 8 channels, Kernel: 3x3, Padding: 1
2.  **Batch Normalization 1 (`bn1`):** Normalizes output of `conv1`.
3.  **ReLU Activation:** Applies ReLU.
4.  **Convolutional Layer 2 (`conv2`):** Input: 8 channels, Output: 16 channels, Kernel: 3x3, Padding: 1
5.  **Batch Normalization 2 (`bn2`):** Normalizes output of `conv2`.
6.  **ReLU Activation:** Applies ReLU.
7.  **Max Pooling 1 (`pool1`):** Kernel: 2x2, Stride: 2
8.  **Convolutional Layer 3 (`conv3`):** Input: 16 channels, Output: 16 channels, Kernel: 3x3, Padding: 1
9.  **Batch Normalization 3 (`bn3`):** Normalizes output of `conv3`.
10. **ReLU Activation:** Applies ReLU.
11. **Convolutional Layer 4 (`conv4`):** Input: 16 channels, Output: 16 channels, Kernel: 3x3, Padding: 1
12. **Batch Normalization 4 (`bn4`):** Normalizes output of `conv4`.
13. **ReLU Activation:** Applies ReLU.
14. **Max Pooling 2 (`pool2`):** Kernel: 2x2, Stride: 2
15. **Flattening:** Reshapes to 1D vector (7 * 7 * 16).
16. **Fully Connected Layer 1 (`fc1`):** Input: 7 * 7 * 16, Output: 32
17. **ReLU Activation:** Applies ReLU.
18. **Dropout (`dropout`):** Dropout Rate: 0.25
19. **Fully Connected Layer 2 (`fc2`):** Input: 32, Output: 10

**Architecture Summary (`FinalNet`)**

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

### Architecture 2: `CloserNet`

To get closer to the target parameter count of 20,000, we designed a second architecture named `CloserNet` by reducing the number of filters in some convolutional layers and the size of the linear layers. This architecture also incorporates Batch Normalization and a single Dropout layer at the end.

Here's a breakdown of the layers:

1.  **Convolutional Layer 1 (`conv1`):** Input: 1 channel, Output: 8 channels, Kernel: 3x3, Padding: 1
2.  **Batch Normalization 1 (`bn1`):** Normalizes output of `conv1`.
3.  **ReLU Activation:** Applies ReLU.
4.  **Convolutional Layer 2 (`conv2`):** Input: 8 channels, Output: 12 channels, Kernel: 3x3, Padding: 1
5.  **Batch Normalization 2 (`bn2`):** Normalizes output of `conv2`.
6.  **ReLU Activation:** Applies ReLU.
7.  **Max Pooling 1 (`pool1`):** Kernel: 2x2, Stride: 2
8.  **Convolutional Layer 3 (`conv3`):** Input: 12 channels, Output: 16 channels, Kernel: 3x3, Padding: 1
9.  **Batch Normalization 3 (`bn3`):** Normalizes output of `conv3`.
10. **ReLU Activation:** Applies ReLU.
11. **Convolutional Layer 4 (`conv4`):** Input: 16 channels, Output: 16 channels, Kernel: 3x3, Padding: 1
12. **Batch Normalization 4 (`bn4`):** Normalizes output of `conv4`.
13. **ReLU Activation:** Applies ReLU.
14. **Max Pooling 2 (`pool2`):** Kernel: 2x2, Stride: 2
15. **Flattening:** Reshapes to 1D vector (7 * 7 * 16).
16. **Fully Connected Layer 1 (`fc1`):** Input: 7 * 7 * 16, Output: 20
17. **ReLU Activation:** Applies ReLU.
18. **Dropout (`dropout`):** Dropout Rate: 0.25
19. **Fully Connected Layer 2 (`fc2`):** Input: 20, Output: 10

**Architecture Summary (`CloserNet`)**

| Layer             | Type          | Output Shape     | Param # |
|-------------------|---------------|------------------|---------|
| Conv2d-1          | Conv2d        | [-1, 8, 28, 28]  | 80      |
| BatchNorm2d-2     | BatchNorm2d   | [-1, 8, 28, 28]  | 16      |
| Conv2d-3          | Conv2d        | [-1, 12, 28, 28] | 876     |
| BatchNorm2d-4     | BatchNorm2d   | [-1, 12, 28, 28] | 24      |
| MaxPool2d-5       | MaxPool2d     | [-1, 12, 14, 14] | 0       |
| Conv2d-6          | Conv2d        | [-1, 16, 14, 14] | 1,744   |
| BatchNorm2d-7     | BatchNorm2d   | [-1, 16, 14, 14] | 32      |
| Conv2d-8          | Conv2d        | [-1, 16, 14, 14] | 2,320   |
| BatchNorm2d-9     | BatchNorm2d   | [-1, 16, 14, 14] | 32      |
| MaxPool2d-10      | MaxPool2d     | [-1, 16, 7, 7]   | 0       |
| Linear-11         | Linear        | [-1, 20]         | 15,700  |
| Dropout-12        | Dropout       | [-1, 20]         | 0       |
| Linear-13         | Linear        | [-1, 10]         | 210     |
| **Total Params**  |               |                  | **21,034**|

This `CloserNet` architecture provides a good balance between model complexity and performance for the MNIST dataset, achieving a high accuracy with a parameter count very close to the desired range of 20,000.
