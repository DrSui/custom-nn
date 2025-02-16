import numpy as np
import cupy as cp
from data import get_data
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from scipy import signal
from resize import ResizeIMG_64_64
#math funcitons
#activation funcitons
def fast_gelu(x):
    return 1 / (1 + cp.exp(-x))

def delta_fast_gelu(x):
    return (x * (1 - x))
def softmax(x):
    relative_x = cp.exp(x-cp.max(x))
    return relative_x / relative_x.sum()
class NeuralNetwork:
    def __init__(self, n, learn_rate, size, kernel_size):
        #set hyper parameters
        self.array = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat", "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit"]
        self.n = n
        self.learn_rate = learn_rate
        self.kernel_filter = cp.random.uniform(-0.5,0.5, (kernel_size,kernel_size,kernel_size))
        #set weights
        entrance_size = (size-kernel_size + 1)**2
        #                                                        (62,62)
        self.weight_input_hidden1 = cp.random.uniform(-0.5, 0.5, (3844,))
        self.weight_hidden1_hidden2 = cp.random.uniform(-0.5, 0.5, (3844,))
        self.weight_hidden2_hidden3 = cp.random.uniform(-0.5, 0.5, (3844, ))
        self.weight_hidden3_output = cp.random.uniform(-0.5, 0.5, (11,3844))
        print(self.weight_hidden3_output.shape)
        #set biases
        self.bias_input_hidden1 = cp.zeros(3844, )
        self.bias_hidden1_hidden2 = cp.zeros(3844, )
        self.bias_hidden2_hidden3 = cp.zeros(3844, )
        self.bias_hidden3_output = cp.zeros(11, )
        self.bias_convolved = cp.zeros((62,62,1))
        
    def forward(self, x):
        print("here")
        print(x.shape)
        print(self.kernel_filter.shape)
        self.convolved_unbias = cp.array(signal.fftconvolve(x.get(),self.kernel_filter.get(),mode="valid"))
        self.convolved = self.convolved_unbias + self.bias_convolved
        print(self.convolved.shape)
        self.convolved_flat = fast_gelu(self.convolved.reshape(3844,))
        # Forward propagation input -> hidden1
        hidden1_pre = self.bias_input_hidden1 + self.weight_input_hidden1 * self.convolved_flat
        self.hidden1 = fast_gelu(hidden1_pre)
        print(self.hidden1.shape, "HIDDEN SHAPE")
        # Forward propagation hidden1 -> hidden2
        hidden2_pre = self.bias_hidden1_hidden2 + self.weight_hidden1_hidden2 * self.hidden1
        self.hidden2 = fast_gelu(hidden2_pre)

        # Forward propagation hidden2 -> hidden3
        print(self.weight_hidden2_hidden3.shape, self.hidden2.shape)
        hidden3_pre = self.bias_hidden2_hidden3 + self.weight_hidden2_hidden3 * self.hidden2
        self.hidden3 = fast_gelu(hidden3_pre)
        print(self.weight_hidden3_output.shape, self.hidden3.shape,"GRRRRRRRRRRRR") 
        # Forward propagation hidden2 -> output
        self.output_pre = self.bias_hidden3_output + (self.weight_hidden3_output @ self.hidden3)
        print(self.output_pre)
        self.output = softmax(self.output_pre)
        print("HERHERKJFHSDAKFJHSADFHSADLKF2")
        print(self.output.shape)
        return self.output
    
    def backward(self, img, output, label):
        # Convert label to one-hot encoding if it's not already
        if label.size == 1:
            one_hot_label = cp.zeros(11)
            one_hot_label[int(label)] = 1
        else:
            one_hot_label = label
            
        # Backpropagation output -> hidden3 (softmax cross-entropy gradient)
        # For softmax + cross-entropy, the gradient is simply (output - target)
        delta_output = self.output - one_hot_label  # Shape: (11,)
        
        # Update weights and biases for output layer
        # delta_output shape: (11,)
        # self.hidden3 shape: (3844,)
        grad_weight_hidden3_output = cp.outer(delta_output, self.hidden3)  # Shape: (11, 3844)
        self.weight_hidden3_output -= self.learn_rate * grad_weight_hidden3_output
        self.bias_hidden3_output -= self.learn_rate * delta_output
        
        # Backpropagation hidden3 -> hidden2
        # delta_hidden = (W.T @ delta_next) * f'(z)
        # self.weight_hidden3_output.T shape: (3844, 11)
        # delta_output shape: (11,)
        delta_hidden3 = (self.weight_hidden3_output.T @ delta_output) * delta_fast_gelu(self.hidden3)  # Shape: (3844,)
        
        # self.weight_hidden2_hidden3 shape: (3844,)
        # delta_hidden3 shape: (3844,)
        # self.hidden2 shape: (3844,)
        grad_weight_hidden2_hidden3 = delta_hidden3 * self.hidden2  # Element-wise multiplication
        self.weight_hidden2_hidden3 -= self.learn_rate * grad_weight_hidden2_hidden3
        self.bias_hidden2_hidden3 -= self.learn_rate * delta_hidden3
        
        # Backpropagation hidden2 -> hidden1
        # delta_hidden2 shape: (3844,)
        delta_hidden2 = (self.weight_hidden2_hidden3 * delta_hidden3) * delta_fast_gelu(self.hidden2)
        grad_weight_hidden1_hidden2 = delta_hidden2 * self.hidden1  # Element-wise multiplication
        self.weight_hidden1_hidden2 -= self.learn_rate * grad_weight_hidden1_hidden2
        self.bias_hidden1_hidden2 -= self.learn_rate * delta_hidden2
        
        # Backpropagation hidden1 -> convolved
        delta_hidden1 = (self.weight_hidden1_hidden2 * delta_hidden2) * delta_fast_gelu(self.hidden1)
        grad_weight_input_hidden1 = delta_hidden1 * self.convolved_flat  # Element-wise multiplication
        self.weight_input_hidden1 -= self.learn_rate * grad_weight_input_hidden1
        self.bias_input_hidden1 -= self.learn_rate * delta_hidden1
        
        # Backpropagation convolved -> input
        # Reshape delta_hidden1 for convolution
        delta_conv_reshaped = delta_hidden1.reshape(62, 62, 1)
        
        # Correlate for gradient of convolution
        delta_kernel = cp.array(signal.correlate(
            img.reshape(64, 64, 3).get(),
            delta_conv_reshaped.get(),
            mode="valid"
        ))
        
        self.bias_convolved -= self.learn_rate * delta_conv_reshaped
        self.kernel_filter -= self.learn_rate * delta_kernel
        
    def train(self, epochs):
        images, labels = get_data()
        images = cp.asarray(images)
        labels = cp.asarray(labels)
        print("training...")
        total_loss = 0.0
        nr_correct = 0
        total_time = 0
        for epoch in range(epochs):
            shuffled_indices = cp.random.permutation(images.shape[0])
            start_time = time.time()
            for idx in shuffled_indices:
                print("NEW GEN")
                img = images[idx]
                label = labels[idx]
                
                one_hot_label = cp.zeros((11,))
                one_hot_label[label] = 1
                
                img = img.reshape(64,64,3)
                output = self.forward(img)
                
                nr_correct += int(cp.argmax(output) == cp.argmax(one_hot_label))
                
                # Cross-entropy loss for multi-class classification
                loss = -cp.sum(one_hot_label * cp.log(output + 1e-10))
                total_loss += loss
                
                # Call backward with proper arguments
                self.backward(img, output, one_hot_label)
                
            print(len(shuffled_indices))
            # Show accuracy for this epoch
            end_time = time.time()
            delta_time = end_time - start_time
            nr_avg = round((nr_correct / images.shape[0]) * 100, 2)
            avg_loss = total_loss / images.shape[0]
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Average Loss: {avg_loss.item():.4f}")
            print(f"Accuracy: {nr_avg}%")
            print(f"time taken: {delta_time}")
            total_time += delta_time
            nr_correct = 0
        print(f"total time was: {total_time} average time: {total_time/epochs}")
    def save(self):
        cp.savez("neuralNetworkData.npz", 
            weight_input_hidden1=self.weight_input_hidden1, 
            weight_hidden1_hidden2 = self.weight_hidden1_hidden2,
            weight_hidden2_hidden3 = self.weight_hidden2_hidden3,
            weight_hidden3_output= self.weight_hidden3_output,
            bias_input_hidden1 = self.bias_input_hidden1,
            bias_hidden1_hidden2 = self.bias_hidden1_hidden2,
            bias_hidden2_hidden3 = self.bias_hidden2_hidden3,
            bias_hidden3_output= self.bias_hidden3_output)
    def load(self):
        print("loading...")
        data = cp.load("neuralNetworkData.npz")
        
        self.weight_input_hidden1 = cp.asarray(data["weight_input_hidden1"])
        self.weight_hidden1_hidden2 = cp.asarray(data["weight_hidden1_hidden2"])
        self.weight_hidden2_hidden3 = cp.asarray(data["weight_hidden2_hidden3"])
        self.weight_hidden3_output = cp.asarray(data["weight_hidden3_output"])
        
        self.bias_input_hidden1 = cp.asarray(data["bias_input_hidden1"])
        self.bias_hidden1_hidden2 = cp.asarray(data["bias_hidden1_hidden2"])
        self.bias_hidden2_hidden3 = cp.asarray(data["bias_hidden2_hidden3"])
        self.bias_hidden3_output = cp.asarray(data["bias_hidden3_output"])
def main():
    uinput = input("load or train or both: ")
    size = 64
    n = 200
    learn_rate = 0.01
    #path = r"D:\ai_projects\projects\data\training\Dessert\2.jpg"
    # Convert NumPy arrays to CuPy arrays

    neural_net = NeuralNetwork(n, learn_rate, size,3)
    if uinput == "train":
        epochs = 10
        neural_net.train(epochs)
        neural_net.save()

    else:
        neural_net.load()
        if uinput == "both":
            print("begining training...")
            neural_net.train(10)
            neural_net.save()
    while True:
        path = input("enter the directory to an image: ")
        
        img_path = fr"{path}"
        img_resized = ResizeIMG_64_64(img_path).reshape(64,64,3)
        img = mpimg.imread(img_path)
        print("Image shape:", img.shape)
        plt.imshow(img)

        output = neural_net.forward(img_resized)

        plt.title(f"Is it {neural_net.array[cp.argmax(output).item()]} :)")
        print(cp.exp(output)/cp.sum(cp.exp(output)))
        print(cp.argmax(output))
        plt.show()

if __name__ == "__main__":
    main()
