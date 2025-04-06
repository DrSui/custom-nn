import cupy as cp
from scipy.signal import correlate2d, fftconvolve, convolve

class HiddenLayer:
    def __init__(self, height, width, learn_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, l2_lambda=0.01) -> None:
        # Initialize weights and biases
        self.learn_rate = learn_rate
        self.weights = cp.random.randn(height, width) * cp.sqrt(2.0 / width)  # He initialization
        self.bias = cp.random.randn(height, 1) * 0.01
        
        # Adam optimizer parameters
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_weights = cp.zeros_like(self.weights)
        self.v_weights = cp.zeros_like(self.weights)
        self.m_bias = cp.zeros_like(self.bias)
        self.v_bias = cp.zeros_like(self.bias)
        self.t = 0  # Time step for Adam
        
        # L2 regularization parameter
        self.l2_lambda = l2_lambda
        
        # Gradient clipping threshold
        self.clip_threshold = 5.0

    @staticmethod
    def activation(x, alpha=0.01):
        return cp.where(x > 0, x, alpha * x)  # Leaky ReLU

    def derivative_activation(self, x, alpha=0.01):
        return cp.where(x > 0, 1, alpha)  # Derivative of Leaky ReLU
    
    def clip_gradients(self, grad, threshold=None):
        if threshold is None:
            threshold = self.clip_threshold
        norm = cp.linalg.norm(grad)
        if norm > threshold:
            return grad * (threshold / norm)
        return grad

    def normalize(self, x, epsilon=1e-8):
        """Apply layer normalization"""
        mean = cp.mean(x)
        std = cp.std(x)
        return (x - mean) / (std + epsilon)

    def forward(self, in_matrix):
        self.input = in_matrix.reshape((self.weights.T.shape[0], 1))
        self.output_pre = self.weights @ self.input + self.bias
        self.output_pre = self.normalize(self.output_pre)
        self.output = self.activation(self.output_pre)
        return self.output
    
    def backward(self, dA):
        # Increment time step for Adam
        self.t += 1
        
        # Find relevant derivatives
        dZ = self.clip_gradients(dA * self.derivative_activation(self.output_pre))
        dW = self.clip_gradients(cp.dot(dZ, self.input.T))
        dB = dZ
        
        # Add L2 regularization gradient
        dW_with_reg = dW + (self.l2_lambda * self.weights)
        
        # Adam updates for weights
        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * dW_with_reg
        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (dW_with_reg ** 2)
        
        m_weights_corrected = self.m_weights / (1 - self.beta1 ** self.t)
        v_weights_corrected = self.v_weights / (1 - self.beta2 ** self.t)
        
        # Adam updates for bias
        self.m_bias = self.beta1 * self.m_bias + (1 - self.beta1) * dB
        self.v_bias = self.beta2 * self.v_bias + (1 - self.beta2) * (dB ** 2)
        
        m_bias_corrected = self.m_bias / (1 - self.beta1 ** self.t)
        v_bias_corrected = self.v_bias / (1 - self.beta2 ** self.t)
        
        # Update parameters with Adam
        self.weights -= self.learn_rate * m_weights_corrected / (cp.sqrt(v_weights_corrected) + self.epsilon)
        self.bias -= self.learn_rate * m_bias_corrected / (cp.sqrt(v_bias_corrected) + self.epsilon)
        
        # Clip weights and biases to prevent extreme values
        self.weights = self.clip_gradients(self.weights)
        self.bias = cp.clip(self.bias, -1, 1)  # Prevent runaway biases
        
        # Calculate gradient for next layer
        dN = cp.dot(self.weights.T, dZ)
        
        return dN

class OutputLayer(HiddenLayer): 
    def __init__(self, height, width, learn_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, l2_lambda=0.01) -> None:
        super().__init__(height, width, learn_rate, beta1, beta2, epsilon, l2_lambda)

    @staticmethod 
    def activation(x, alpha=0.01):
        x = x - cp.max(x)  # Ensures numerical stability
        exp_x = cp.exp(x)
        return exp_x / cp.sum(exp_x)
        
    def derivative_activation(self, x, alpha=0.01):
        # Not directly used in this implementation since we calculate
        # the gradient of softmax with cross-entropy directly

        return self.output - x
    
    def forward(self, in_matrix):
        self.input = in_matrix
        self.output_pre = self.weights @ in_matrix + self.bias
        self.output_pre = self.normalize(self.output_pre)
        self.output = self.activation(self.output_pre)
        return self.output
    
    def backward(self, dA):
        # Increment time step for Adam
        self.t += 1
        
        # Direct computation of gradient for softmax with cross-entropy
        dZ = self.clip_gradients(self.output - self.derivative_activation(dA))
        dW = self.clip_gradients(cp.dot(dZ, self.input.T))
        dB = cp.sum(dZ, axis=1, keepdims=True)
        
        # Add L2 regularization gradient
        dW_with_reg = dW + (self.l2_lambda * self.weights)
        
        # Adam updates for weights
        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * dW_with_reg
        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (dW_with_reg ** 2)
        
        m_weights_corrected = self.m_weights / (1 - self.beta1 ** self.t)
        v_weights_corrected = self.v_weights / (1 - self.beta2 ** self.t)
        
        # Adam updates for bias
        self.m_bias = self.beta1 * self.m_bias + (1 - self.beta1) * dB
        self.v_bias = self.beta2 * self.v_bias + (1 - self.beta2) * (dB ** 2)
        
        m_bias_corrected = self.m_bias / (1 - self.beta1 ** self.t)
        v_bias_corrected = self.v_bias / (1 - self.beta2 ** self.t)
        
        # Update parameters with Adam
        self.weights -= self.learn_rate * m_weights_corrected / (cp.sqrt(v_weights_corrected) + self.epsilon)
        self.bias -= self.learn_rate * m_bias_corrected / (cp.sqrt(v_bias_corrected) + self.epsilon)
        
        # Clip weights and biases
        self.weights = self.clip_gradients(self.weights)
        self.bias = cp.clip(self.bias, -1, 1)
        
        # Calculate gradient for next layer
        dN = cp.dot(self.weights.T, dZ)
        
        return dN

class ConvLayer:
    def __init__(self, input_channels, output_channels, kernel_size, size, stride=1, padding=0, 
                learn_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, l2_lambda=0.01):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.learn_rate = learn_rate
        self.entrance_size = size - kernel_size + 1
        
        # Xavier initialization of weights
        self.weights = cp.random.randn(output_channels, input_channels, kernel_size, kernel_size) * cp.sqrt(1. / (input_channels * kernel_size * kernel_size))
        self.biases = cp.zeros((output_channels, 1))
        
        # Adam optimizer parameters
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_weights = cp.zeros_like(self.weights)
        self.v_weights = cp.zeros_like(self.weights)
        self.m_biases = cp.zeros_like(self.biases)
        self.v_biases = cp.zeros_like(self.biases)
        self.t = 0  # Time step for Adam
        
        # L2 regularization parameter
        self.l2_lambda = l2_lambda
        
        # Gradient clipping threshold
        self.clip_threshold = 5.0
    
    def pad_input(self, x):
        if self.padding > 0:
            return cp.pad(x, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        return x
    
    def normalize(self, x, epsilon=1e-8):
        """Apply feature map normalization"""
        # Normalize across spatial dimensions for each feature map
        mean = cp.mean(x, axis=(0, 1), keepdims=True)
        std = cp.std(x, axis=(0, 1), keepdims=True)
        return (x - mean) / (std + epsilon)
    
    def clip_gradients(self, grad, threshold=None):
        if threshold is None:
            threshold = self.clip_threshold
        norm = cp.linalg.norm(grad)
        if norm > threshold:
            return grad * (threshold / norm)
        return grad
    
    def forward(self, x):
        self.input = x  # Store input for backpropagation
        height, width, _ = x.shape
        padded_x = self.pad_input(x)
        output_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        output = cp.zeros((output_height, output_width, self.output_channels))
        
        for oc in range(self.output_channels):
            conv_result = cp.zeros((output_height, output_width))
            for ic in range(self.input_channels):
                conv_result += cp.array(convolve(padded_x[:, :, ic].get(), self.weights[oc, ic].get(), mode='valid')[::self.stride, ::self.stride])
            output[:, :, oc] = conv_result + self.biases[oc]
        
        # Apply normalization
        output = self.normalize(output)
        
        return output
    
    def backward(self, d_output):
        # Increment time step for Adam
        self.t += 1
        
        padded_input = self.pad_input(self.input)
        d_padded_input = cp.zeros_like(padded_input)
        d_output = d_output.reshape(self.entrance_size, self.entrance_size, self.output_channels)
        
        grad_weights = cp.zeros_like(self.weights)
        grad_biases = cp.zeros_like(self.biases)
        
        for oc in range(self.output_channels):
            grad_biases[oc] += cp.sum(d_output[:, :, oc])
            for ic in range(self.input_channels):
                grad_weights[oc, ic] += cp.array(correlate2d(padded_input[:, :, ic].get(), d_output[:, :, oc].get(), mode='valid'))
                d_padded_input[:, :, ic] += cp.array(convolve(d_output[:, :, oc].get(), self.weights[oc, ic].get(), mode='full'))
        
        # Add L2 regularization gradient
        grad_weights_with_reg = grad_weights + (self.l2_lambda * self.weights)
        
        # Clip gradients
        grad_weights_with_reg = self.clip_gradients(grad_weights_with_reg)
        grad_biases = self.clip_gradients(grad_biases)
        
        # Adam updates for weights
        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * grad_weights_with_reg
        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (grad_weights_with_reg ** 2)
        
        m_weights_corrected = self.m_weights / (1 - self.beta1 ** self.t)
        v_weights_corrected = self.v_weights / (1 - self.beta2 ** self.t)
        
        # Adam updates for biases
        self.m_biases = self.beta1 * self.m_biases + (1 - self.beta1) * grad_biases
        self.v_biases = self.beta2 * self.v_biases + (1 - self.beta2) * (grad_biases ** 2)
        
        m_biases_corrected = self.m_biases / (1 - self.beta1 ** self.t)
        v_biases_corrected = self.v_biases / (1 - self.beta2 ** self.t)
        
        # Update parameters with Adam
        self.weights -= self.learn_rate * m_weights_corrected / (cp.sqrt(v_weights_corrected) + self.epsilon)
        self.biases -= self.learn_rate * m_biases_corrected / (cp.sqrt(v_biases_corrected) + self.epsilon)
        
        # Clip weights to prevent extreme values
        self.weights = self.clip_gradients(self.weights)
        self.biases = cp.clip(self.biases, -1, 1)
        
        # Remove padding from d_padded_input to get d_input
        if self.padding > 0:
            d_input = d_padded_input[self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            d_input = d_padded_input
        
        return cp.array(d_input)
