import numpy as np
import random
import os

class Network:
    def __init__(self, sizes, model_path=None):
        """
        sizes: list of layer sizes, e.g., [784, 128, 64, 10]
        model_path: optional path to load pretrained weights
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        
        # Initialize weights and biases
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        # Load model if path exists
        if model_path is not None:
            path = os.path.abspath(os.path.expanduser(model_path))
            if os.path.exists(path):
                data = np.load(path, allow_pickle=True)
                self.weights = list(data['weights'])
                self.biases = list(data['biases'])
                print(f"Model loaded from: {path}")
            else:
                print(f"Model path {path} not found. Initialized randomly.")

    
    def initialize_weights_biases(self):
        """He initialization for ReLU"""
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) * np.sqrt(2 / x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        print("Weights and biases initialized using He initialization.")
        
    def feedforward(self, a):
        """Compute output of network given input a"""
        # Hidden layers with ReLU
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = relu(np.dot(w, a) + b)
        # Output layer with Softmax
        z = np.dot(self.weights[-1], a) + self.biases[-1]
        return softmax(z)

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train network using mini-batch SGD"""
        n = len(training_data)
        if test_data:
            n_test = len(test_data)
        
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch, eta):
        """Update weights and biases using backpropagation"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases  = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return gradients for weights and biases"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Forward pass
        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = relu(z)
            activations.append(activation)

        # Output layer
        z = np.dot(self.weights[-1], activation) + self.biases[-1]
        zs.append(z)
        activation = softmax(z)
        activations.append(activation)

        # Backward pass
        # For softmax + cross-entropy, derivative simplifies
        delta = activations[-1] - y
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        # Backprop for hidden layers
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = relu_prime(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return number of correctly classified inputs"""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def save(self, filename='digit_model.npz'):
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        np.savez_compressed(save_path, 
                            weights=np.array(self.weights, dtype=object),
                            biases=np.array(self.biases, dtype=object))
        print(f"Model saved to: {save_path}")

    def load(self, filename='model.npz'):
        path = os.path.join('model', filename)
        data = np.load(path, allow_pickle=True)
        self.weights = list(data['weights'])
        self.biases = list(data['biases'])


# --- Activation Functions ---
def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z))  # numerical stability
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)
