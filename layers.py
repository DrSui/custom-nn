import numpy as cp

class HiddenLayer:
    def __init__(self, height, width, learn_rate) -> None:
        #self.size = size
        self.learn_rate = learn_rate
        self.weights = cp.random.randn(height, width) * cp.sqrt(2.0 / width)
        self.bias = cp.random.randn(height,1) * 0.001

    @staticmethod
    def activation(x, alpha=0.01):
        return cp.where(x > 0, x, alpha * x)
    @staticmethod
    def derivative_activation(x,alpha=0.01):
        return cp.where(x > 0, 1, alpha)
    @staticmethod
    def clip_gradients(grad, threshold=5.0):
        norm = cp.linalg.norm(grad)
        if norm > threshold:
            return grad * (threshold / norm)
        return grad

    def forward(self, in_matrix):
        self.input = in_matrix
        self.output_pre = self.weights @ in_matrix + self.bias
        self.output_pre = (self.output_pre - cp.mean(self.output_pre)) / (cp.std(self.output_pre) + 1e-8)
        self.output = self.activation(self.output_pre)
        return self.output
    
    def backward(self, dA):
        # Find relevant derivatives
        dZ = self.clip_gradients(dA * self.derivative_activation(self.output_pre))
        dW = self.clip_gradients(cp.dot(dZ, self.input.T))
        #dB = cp.sum(dZ, axis=1, keepdims=True)
        dN = cp.dot(self.weights.T, dZ)
        
        # Update parameters
        self.weights -= self.learn_rate *self.clip_gradients(dW)
        self.bias -= self.learn_rate * dZ
        self.bias = cp.clip(self.bias, -1, 1)  # Prevent runaway biases
        self.weights = self.clip_gradients(self.weights)
        print(f"Layer gradient mean: {cp.mean(dW):.5f}, min: {cp.min(dW):.5f}, max: {cp.max(dW):.5f}")
        return dN

class OutputLayer(HiddenLayer):
    def __init__(self, height, width, learn_rate=0.01) -> None:
        super().__init__(height, width, learn_rate)
    @staticmethod 
    def softmax(x):
        x = x - cp.max(x)  # Ensures numerical stability
        exp_x = cp.exp(x)
        return exp_x / cp.sum(exp_x)
        
    @staticmethod
    def derivative_activation(x):
        # Not directly used in this implementation since we calculate
        # the gradient of softmax with cross-entropy directly
        return 1
    
    def forward(self, in_matrix):
        self.input = in_matrix
        self.output_pre = self.weights @ in_matrix + self.bias
        self.output_pre = (self.output_pre - cp.mean(self.output_pre)) / (cp.std(self.output_pre) + 1e-8)
        self.output = self.softmax(self.output_pre)
        print("SELF OUT SHAPE:", self.output.shape)
        return self.output
    
    def backward(self, y_true):
        # Direct computation of gradient for softmax with cross-entropy
        # Assumes y_true is one-hot encoded with same shape as output
        dZ = self.clip_gradients(self.output - y_true)
        print(dZ.shape)
        dW = self.clip_gradients(cp.dot(dZ, self.input.T))
        dB = cp.sum(dZ, axis=1, keepdims=True)
        dN = cp.dot(self.weights.T, dZ)
        
        # Update parameters
        self.weights -= self.learn_rate * dW
        self.weights = self.clip_gradients(self.weights)
        self.bias -= self.learn_rate * dB
        self.bias = cp.clip(self.bias, -1, 1)  # Prevent runaway biases
        print(dN.shape)
        return dN
