#!/usr/bin/env python3
# CNN Concepts and Implementations - Q&A as Comments
# Format: #number. Question then Solution: answer

#1. What is a Convolutional Neural Network (CNN), and why is it used for image processing?
# Solution: A CNN is a deep learning architecture that uses convolution operations to automatically learn spatial hierarchies of features. It's used for image processing because it preserves spatial relationships, reduces parameters via weight sharing, and is translation invariant.

#2. What are the key components of a CNN architecture?
# Solution: Convolutional layers, activation functions (ReLU), pooling layers, fully connected layers, and optional batch normalization & dropout.

#3. What is the role of the convolutional layer in CNNs?
# Solution: It applies filters (kernels) to the input to extract feature maps by sliding the filter across the spatial dimensions and computing dot products.

#4. What is a filter (kernel) in CNNs?
# Solution: A small matrix of learnable weights that slides over the input image to detect specific features like edges, textures, or patterns.

#5. What is pooling in CNNs, and why is it important?
# Solution: Pooling reduces the spatial dimensions of feature maps, making the model more computationally efficient and providing translation invariance.

#6. What are the common types of pooling used in CNNs?
# Solution: Max pooling (takes maximum value in each window) and average pooling (takes average value).

#7. How does the backpropagation algorithm work in CNNs?
# Solution: It computes gradients of the loss with respect to each weight by applying the chain rule. For convolutional layers, gradients are calculated using transposed convolutions; for pooling layers, gradients are routed to the max position (max pooling) or evenly distributed (average pooling).

#8. What is the role of activation functions in CNNs?
# Solution: They introduce non‑linearity, allowing CNNs to learn complex patterns. ReLU is most common because it is fast and mitigates vanishing gradients.

#9. What is the concept of receptive fields in CNNs?
# Solution: The region in the input image that a particular feature (neuron) in a deeper layer “sees”. Stacking layers increases the receptive field without losing resolution.

#10. Explain the concept of tensor space in CNNs.
# Solution: Tensors in CNNs are multi‑dimensional arrays (height, width, channels, batch). The tensor space refers to how data transforms through layers – from raw pixel tensor to high‑level feature tensors.

#11. What is LeNet-5, and how does it contribute to the development of CNNs?
# Solution: LeNet-5 (1998) was one of the first CNNs for handwritten digit recognition. It introduced the basic pattern: conv → pool → conv → pool → fully connected.

#12. What is AlexNet, and why was it a breakthrough in deep learning?
# Solution: AlexNet (2012) won ImageNet by using ReLU activations, dropout, data augmentation, and GPU training. It showed deep CNNs can outperform traditional computer vision methods.

#13. What is VGGNet, and how does it differ from AlexNet?
# Solution: VGGNet uses only 3x3 convolutional filters stacked deeply (16‑19 layers). It is simpler and more uniform than AlexNet, but computationally heavier.

#14. What is GoogLeNet, and what is its main innovation?
# Solution: GoogLeNet (Inception) introduced the Inception module – parallel convolutions (1x1, 3x3, 5x5) and pooling, then concatenated. This reduces parameters and improves efficiency.

#15. What is ResNet, and what problem does it solve?
# Solution: ResNet uses skip connections (residual blocks) to solve the vanishing gradient problem, allowing training of very deep networks (152+ layers).

#16. What is DenseNet, and how does it differ from ResNet?
# Solution: DenseNet connects each layer to every other layer in a feed‑forward fashion (dense blocks). It encourages feature reuse, reduces parameters, and improves gradient flow compared to ResNet.

#17. What are the main steps involved in training a CNN from scratch?
# Solution: Data collection & preprocessing, defining architecture, initializing weights, forward pass, loss computation, backpropagation, optimizer update, and iterative training over epochs.



#PRACTICAL

#1. Implement a basic convolution operation using a filter and a 5x5 image (matrix).

import numpy as np
image = np.array([[1,2,3,4,5], [5,4,3,2,1], [1,2,3,4,5], [5,4,3,2,1], [1,2,3,4,5]])
kernel = np.array([[1,0,-1], [1,0,-1], [1,0,-1]])
output = np.zeros((3,3))
for i in range(3):
     for j in range(3):
         output[i,j] = np.sum(image[i:i+3, j:j+3] * kernel)

#2. Implement max pooling on a 4x4 feature map with a 2x2 window.

import numpy as np
feature_map = np.random.rand(4,4)
pooled = np.zeros((2,2))
for i in range(0,4,2):
     for j in range(0,4,2):
         pooled[i//2, j//2] = np.max(feature_map[i:i+2, j:j+2])

#3. Implement the ReLU activation function on a feature map.

def relu(x):
     return np.maximum(0, x)
feature_map = np.random.randn(5,5)
activated = relu(feature_map)

#4. Create a simple CNN model with one convolutional layer and a fully connected layer, using random data.

import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
random_data = tf.random.normal((32,28,28,1))
output = model(random_data)

#5. Generate a synthetic dataset using random noise and train a simple CNN model on it.

X_train = np.random.randn(1000, 32,32,1)
y_train = np.random.randint(0, 10, size=(1000,))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, verbose=0)

#6. Create a simple CNN using Keras with one convolution layer and a max-pooling layer.

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2))
])

#7. Write code to add a fully connected layer after the convolution and max-pooling layers in a CNN.

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

#8. Write code to add batch normalization to a simple CNN model.

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), input_shape=(28,28,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D((2,2))
])

#9. Write code to add dropout regularization to a simple CNN model.

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

#10. Write code to print the architecture of the VGG16 model in Keras.

from tensorflow.keras.applications import VGG16
model = VGG16(weights=None, input_shape=(224,224,3))
model.summary()

#11. Write code to plot the accuracy and loss graphs after training a CNN model.

import matplotlib.pyplot as plt
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, verbose=0)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend(); plt.show()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(); plt.show()

#12. Write code to print the architecture of the ResNet50 model in Keras.

from tensorflow.keras.applications import ResNet50
model = ResNet50(weights=None, input_shape=(224,224,3))
model.summary()

#13. Write code to train a basic CNN model and print the training loss and accuracy after each epoch.

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, verbose=1)
for epoch, (loss, acc) in enumerate(zip(history.history['loss'], history.history['accuracy'])):
    print(f"Epoch {epoch+1}: loss = {loss:.4f}, accuracy = {acc:.4f}")