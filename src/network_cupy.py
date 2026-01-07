import cupy as cp
import random
import os

class Network(object):
    def __init__(self, sizes, model_path=None):
        self.num_layers = len(sizes)
        self.sizes = sizes
        
        # Initialize biases as a list
        self.biases = [cp.random.randn(y, 1) for y in sizes[1:]]
        
        # Initialize weights as a list
        self.weights = [cp.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        # If model_path is provided, try to load weights and biases
        if model_path is not None:
            path = os.path.abspath(os.path.expanduser(model_path))
            if os.path.exists(path):
                data = cp.load(path, allow_pickle=True)
                self.weights = list(data['weights'])
                self.biases = list(data['biases'])
                print(f"Model loaded from: {path}")
            else:
                print(f"Model path {path} not found. Initialized randomly.")

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(cp.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
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
        nabla_b = [cp.zeros(b.shape) for b in self.biases]
        nabla_w = [cp.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [cp.zeros(b.shape) for b in self.biases]
        nabla_w = [cp.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = cp.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = cp.dot(delta, activations[-2].T)

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = cp.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = cp.dot(delta, activations[-l-1].T)

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(cp.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return output_activations - y

    def save(self, filename='digit_model.npz'):
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        cp.savez_compressed(save_path, 
                            weights=cp.array(self.weights, dtype=object),
                            biases=cp.array(self.biases, dtype=object))
        print(f"Model saved to: {save_path}")

    def load(self, filename='model.npz'):
        path = os.path.join('model', filename)
        data = cp.load(path, allow_pickle=True)
        self.weights = list(data['weights'])
        self.biases = list(data['biases'])

# Sigmoid functions
def sigmoid(z):
    return 1.0 / (1.0 + cp.exp(-z))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)
