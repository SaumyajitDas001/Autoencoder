import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
train_data = pd.read_csv('mnist_train.csv')
test_data = pd.read_csv('mnist_test.csv')

train_labels = train_data['label']
test_labels = test_data['label']

train_data = train_data.drop('label', axis=1).values.reshape(-1, 784)
test_data = test_data.drop('label', axis=1).values.reshape(-1, 784)

train_data = train_data / 255.0
test_data = test_data / 255.0

# Activation functions and loss functions
def relu(x):
    return np.maximum(0.01 * x, x)  # Leaky ReLU

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def bce_loss(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Optimizer: Adam
class AdamOptimizer:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

    def update(self, param, grad, m, v):
        self.t += 1
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return param, m, v

# Dense layer operations
def dense(x, w, b):
    return np.dot(x, w.T) + b.T

# Encoder class
class Encoder:
    def __init__(self, input_size, hidden_size, optimizer):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.optimizer = optimizer
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.m_w1 = np.zeros_like(self.W1)
        self.v_w1 = np.zeros_like(self.W1)
        self.m_b1 = np.zeros_like(self.b1)
        self.v_b1 = np.zeros_like(self.b1)

    def forward(self, x):
        self.x = x
        self.z1 = dense(x, self.W1, self.b1)
        self.a1 = relu(self.z1)
        return self.a1

    def backward(self, grad):
        dz1 = grad * (self.z1 > 0)
        self.dW1 = np.dot(dz1.T, self.x) / self.x.shape[0]
        self.db1 = np.sum(dz1, axis=0, keepdims=True).T / self.x.shape[0]
        return np.dot(dz1, self.W1)

    def update(self):
        self.W1, self.m_w1, self.v_w1 = self.optimizer.update(self.W1, self.dW1, self.m_w1, self.v_w1)
        self.b1, self.m_b1, self.v_b1 = self.optimizer.update(self.b1, self.db1, self.m_b1, self.v_b1)

# Decoder class
class Decoder:
    def __init__(self, hidden_size, output_size, optimizer):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.optimizer = optimizer
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((output_size, 1))
        self.m_w2 = np.zeros_like(self.W2)
        self.v_w2 = np.zeros_like(self.W2)
        self.m_b2 = np.zeros_like(self.b2)
        self.v_b2 = np.zeros_like(self.b2)

    def forward(self, x):
        self.x = x
        self.z2 = dense(x, self.W2, self.b2)
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, grad):
        self.dz2 = grad * self.a2 * (1 - self.a2)
        self.dW2 = np.dot(self.dz2.T, self.x) / self.x.shape[0]
        self.db2 = np.sum(self.dz2, axis=0, keepdims=True).T / self.x.shape[0]
        return np.dot(self.dz2, self.W2)

    def update(self):
        self.W2, self.m_w2, self.v_w2 = self.optimizer.update(self.W2, self.dW2, self.m_w2, self.v_w2)
        self.b2, self.m_b2, self.v_b2 = self.optimizer.update(self.b2, self.db2, self.m_b2, self.v_b2)

# Autoencoder class
class Autoencoder:
    def __init__(self, input_size, hidden_size, optimizer):
        self.encoder = Encoder(input_size, hidden_size, optimizer)
        self.decoder = Decoder(hidden_size, input_size, optimizer)

    def forward(self, x):
        encoded = self.encoder.forward(x)
        decoded = self.decoder.forward(encoded)
        return decoded

    def backward(self, x, y):
        decoded = self.forward(x)
        grad = decoded - y
        decoder_grad = self.decoder.backward(grad)
        self.decoder.update()
        encoder_grad = self.encoder.backward(decoder_grad)
        self.encoder.update()

    def train(self, x, y, epochs, batch_size):
        for epoch in range(epochs):
            for i in range(0, x.shape[0], batch_size):
                x_batch = x[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                self.backward(x_batch, y_batch)

# Training the Autoencoder
learning_rate = 0.001
batch_size = 64
epochs = 10
optimizer = AdamOptimizer(lr=learning_rate)
autoencoder = Autoencoder(784, 256, optimizer)
autoencoder.train(train_data, train_data, epochs, batch_size)

# Add evaluate and predict methods to Autoencoder
class Autoencoder:
    # Previous methods ...

    def evaluate(self, x, y):
        """Calculate the loss on the provided dataset."""
        decoded = self.forward(x)
        return bce_loss(y, decoded)

    def predict(self, x):
        """Predict reconstructed output for given input."""
        return self.forward(x)

# Testing the model
test_loss = autoencoder.evaluate(test_data, test_data)
print(f'Test Loss: {test_loss}')

n = 10  # Number of images to display
plt.figure(figsize=(20, 6))
for i in range(n):
    # Original Image
    ax = plt.subplot(3, n, i + 1)
    plt.title("Original")
    plt.imshow(test_data[i].reshape(28, 28), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Encoded Image
    encoded_data = autoencoder.encoder.forward(test_data)
    ax = plt.subplot(3, n, i + 1 + n)
    plt.title("Encoded")
    # Reshape the encoded data to a flat grid (e.g., 16x16 for hidden_size=256)
    ax.imshow(encoded_data[i].reshape(-1, 16), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Reconstructed Image
    reconstructed_data = autoencoder.predict(test_data)
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.title("Reconstructed")
    plt.imshow(reconstructed_data[i].reshape(28, 28), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.tight_layout()
plt.show()




































































































































































































































































































































































































































































































































2




















