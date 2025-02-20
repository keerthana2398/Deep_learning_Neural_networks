# Deep_learning_Neural_networks
# Artificial Neural Networks (ANN)

## 📖 Introduction  
Deep Learning is a subset of Machine Learning that uses **Artificial Neural Networks (ANNs)** to mimic the way the human brain processes data. It is widely used in fields like **computer vision, natural language processing, and autonomous systems**.

---

## 🧠 **1. Understanding Artificial Neural Networks (ANN)**  
An **Artificial Neural Network (ANN)** consists of multiple layers of **neurons** that process information. The main components of an ANN are:  
- **Input Layer**: Takes in features as input.  
- **Hidden Layers**: Perform computations using weights and activation functions.  
- **Output Layer**: Produces the final prediction.  

### **🔗 Forward Propagation (Feedforward Process)**
Forward propagation is the process where inputs are passed **through the network** to compute the output.

#### **Mathematical Formulation**  
For a given neuron:  
\[
Z = W \cdot X + B
\]
\[
A = f(Z)
\]
Where:  
- \( W \) = Weights  
- \( X \) = Inputs  
- \( B \) = Bias  
- \( f(Z) \) = Activation function  

---

### **🔁 Backpropagation (Gradient Descent)**
Backpropagation is a technique to optimize the weights of the neural network by **minimizing the error** using the gradient descent algorithm.

#### **Steps of Backpropagation**  
1. **Compute the loss** using a loss function like Mean Squared Error (MSE) or Cross-Entropy.  
2. **Calculate the gradients** using differentiation (Chain Rule).  
3. **Update weights** using the gradient descent formula:  
   \[
   W = W - \alpha \frac{\partial Loss}{\partial W}
   \]
   where \( \alpha \) is the learning rate.  

---

## ⚡ **2. Activation Functions in Deep Learning**
Activation functions add **non-linearity** to the network, enabling it to learn complex patterns.

### ✅ Common Activation Functions  
| Function | Formula | Use Case |
|----------|---------|----------|
| **Sigmoid** | \( f(x) = \frac{1}{1 + e^{-x}} \) | Binary classification |
| **ReLU** | \( f(x) = \max(0, x) \) | Most hidden layers |
| **Tanh** | \( f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \) | Better than Sigmoid |
| **Softmax** | \( f(x_i) = \frac{e^{x_i}}{\sum e^{x_j}} \) | Multi-class classification |

---

## 🏎️ **3. Optimizers in Deep Learning**
Optimizers improve model training by updating weights efficiently.

### **🚀 Popular Optimizers**
- **SGD (Stochastic Gradient Descent)**: Basic optimizer, slower convergence.  
- **Adam (Adaptive Moment Estimation)**: Combines momentum and adaptive learning rate, widely used.  
- **RMSprop**: Best for recurrent neural networks (RNNs).  

---

## 🏗️ **4. Deep Learning Architectures**
### **1️⃣ Artificial Neural Networks (ANN)**
- Fully connected networks.
- Used for tabular data and structured datasets.

### **2️⃣ Convolutional Neural Networks (CNN)**
- Specialized for image processing.
- Uses convolutional layers to detect spatial features.

### **3️⃣ Recurrent Neural Networks (RNN)**
- Designed for sequence data like text or time series.
- Uses recurrent connections to retain memory.

### **4️⃣ Long Short-Term Memory (LSTM)**
- Advanced RNN that prevents vanishing gradient issues.
- Used in NLP and speech recognition.

---

## 🛠️ **5. ANN Implementation with Python (Using Built-in Dataset)**
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
digits = load_digits()
X = digits.data
y = digits.target

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_categorical = keras.utils.to_categorical(y, num_classes=10)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

# Build ANN model
model = Sequential([
    Dense(64, activation='relu', input_shape=(64,)),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")



import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset (handwritten digits)
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalize the dataset (scale pixel values between 0 and 1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Flatten the images (28x28 -> 784 pixels)
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

# Define the ANN model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),  # Hidden layer 1
    keras.layers.Dense(64, activation='relu'),  # Hidden layer 2
    keras.layers.Dense(10, activation='softmax')  # Output layer (10 classes)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Plot accuracy and loss
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')

plt.show()
# 🧠 Deep Learning - Convolutional Neural Networks (CNN) & Recurrent Neural Networks (RNN)

## 📖 Introduction  
Deep Learning models are categorized into different architectures depending on the nature of data.  
- **CNN (Convolutional Neural Networks)** are specialized for **image processing**.  
- **RNN (Recurrent Neural Networks)** are designed for **sequence-based data** like text and time series.

---

# 🖼️ **1. Convolutional Neural Networks (CNN)**
CNNs are mainly used for **image recognition, classification, and object detection**. They reduce the number of parameters compared to fully connected networks by using **convolutions**.

## 🔗 **1.1 CNN Architecture**
A CNN consists of multiple layers:
1. **Convolutional Layer** - Extracts features using filters.
2. **Activation Function** (e.g., ReLU) - Introduces non-linearity.
3. **Pooling Layer** - Reduces the spatial size of the feature maps.
4. **Fully Connected Layer** - Processes extracted features for classification.

## 🏗️ **1.2 Components of CNN**
### ✅ **1. Convolutional Layer**
- Applies filters (kernels) to extract patterns like edges, textures.
- Formula:
  \[
  Z = W \ast X + B
  \]
  where \( \ast \) represents convolution.

### ✅ **2. Activation Function**
- **ReLU (Rectified Linear Unit)** is commonly used.
- Formula:
  \[
  f(x) = \max(0, x)
  \]

### ✅ **3. Pooling Layer**
- **Max Pooling** reduces spatial size, keeping the most significant features.

- Example:[[2, 3, 1, 0], → [[3, 1], [4, 6, 5, 2], [8, 9]] [8, 9, 6, 3], [1, 2, 0, 1]]


### ✅ **4. Fully Connected Layer**
- Combines extracted features to make predictions.

---

## 🛠️ **1.3 CNN Implementation in Python**
```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess data
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# Build CNN model
model = models.Sequential([
  layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
  layers.MaxPooling2D((2,2)),
  layers.Conv2D(64, (3,3), activation='relu'),
  layers.MaxPooling2D((2,2)),
  layers.Flatten(),
  layers.Dense(64, activation='relu'),
  layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
2. Recurrent Neural Networks (RNN)
RNNs are used for sequential data, such as time series forecasting, speech recognition, and NLP.

🔗 2.1 RNN Architecture
RNNs maintain a memory state by passing the previous output as input to the next step.

✅ How RNN Works
At time step 
𝑡
t:

ℎ
𝑡
=
𝑓
(
𝑊
⋅
𝑋
𝑡
+
𝑈
⋅
ℎ
𝑡
−
1
+
𝐵
)
h 
t
​
 =f(W⋅X 
t
​
 +U⋅h 
t−1
​
 +B)
where:

𝑋
𝑡
X 
t
​
  = Input at time 
𝑡
t.
ℎ
𝑡
−
1
h 
t−1
​
  = Hidden state from previous time step.
𝑊
,
𝑈
,
𝐵
W,U,B = Learnable weights.
🏗️ 2.2 Types of RNNs
RNN Type	Description
Vanilla RNN	Basic recurrent neural network.
LSTM (Long Short-Term Memory)	Handles long-term dependencies and prevents vanishing gradients.
GRU (Gated Recurrent Unit)	A simplified version of LSTM with fewer parameters.
🛠️ 2.3 RNN Implementation in Python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# Load dataset (IMDB Movie Reviews)
max_features = 10000
maxlen = 100

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences to ensure equal length
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# Build RNN model
model = Sequential([
    Embedding(input_dim=max_features, output_dim=32),
    SimpleRNN(64, return_sequences=False),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
3. CNN vs. RNN - When to Use?
Feature	CNN	RNN
Data Type	Images, spatial data	Sequences, text, time series
Layer Type	Convolutional + pooling	Recurrent units (LSTM, GRU)
Memory	No memory of past inputs	Retains past information
Application	Image recognition, object detection	Speech, NLP, forecasting
🎯 Conclusion
CNNs are best for image-based applications.
RNNs excel in sequential data processing.
LSTMs and GRUs improve RNN performance by addressing long-term dependencies.
