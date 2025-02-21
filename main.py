import numpy as cp
from data import get_data
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from scipy import signal
from layers import HiddenLayer, OutputLayer
from resize import ResizeIMG_64_64
#math funcitons
#activation funcitons
def fast_gelu(x, alpha=0.01):
    return cp.where(x > 0, x, alpha * x)
def delta_fast_gelu(x, alpha=0.01):
    return cp.where(x > 0, 1, alpha)
class NeuralNetwork:
    def __init__(self, n, learn_rate, size, kernel_size):
        #set hyper parameters
        self.array = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat", "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit"]
        self.n = n
        self.learn_rate = learn_rate
        #self.kernel_filter = cp.random.uniform(-0.5,0.5, (kernel_size,kernel_size,kernel_size))
        self.kernel_filter = cp.random.randn(kernel_size, kernel_size, kernel_size) * cp.sqrt(2.0/kernel_size)
        #set weights
        self.entrance_size = (size-kernel_size + 1)**2
        #                                                        (62,62)
        self.layers = [
            HiddenLayer(128, 3844, learn_rate),
            #HiddenLayer(11, 11, learn_rate),
            #HiddenLayer(11, 11, learn_rate),
            OutputLayer(11, 128, learn_rate)
        ]
        self.outputs = []
        self.gradients = []
        self.bias_convolved = cp.random.randn(62,62,1) * 0.001
    def forward(self, x):
        print("---------------------------FORWARD------------------------------")
        self.convolved_unbias = cp.array(signal.fftconvolve(x,self.kernel_filter,mode="valid"))
        self.convolved = self.convolved_unbias + self.bias_convolved
        self.convolved = (self.convolved - cp.mean(self.convolved)) / (cp.std(self.convolved) + 1e-8)
        self.outputs.append(fast_gelu(self.convolved.reshape(self.entrance_size,1)))
        for layer_index in range(len(self.layers)):
            print(layer_index)
            self.outputs.append(self.layers[layer_index].forward(self.outputs[layer_index]))
        #print(self.outputs[0])
        
        # Forward propagation input -> hidden1
        print("-----------------------OUTPUT---------------------------------------")
        print(self.outputs[-1][0][0])
        num_zeros = cp.sum(self.outputs[-1] == 0)
        total_neurons = self.outputs[-1].size
        zero_percentage = (num_zeros / total_neurons) * 100
        print(f"Layer {len(self.outputs)-1}: {zero_percentage:.2f}% neurons are zero")
        return self.outputs[-1].reshape(11,1)
    
    def backward(self, img, output, label):
        print("--------------------------------------BACKWARD-----------------------------------------")
        # Convert label to one-hot encoding if it's not already
        if label.size == 1:
            one_hot_label = cp.zeros(11)
            one_hot_label[int(label)] = 1
        else:
            one_hot_label = label
            
        # Backpropagation output -> hidden3 (softmax cross-entropy gradient)
        # For softmax + cross-entropy, the gradient is simply (output - target)
        self.gradients = [[],[],[],[]]
        print(len(self.gradients))
        print(one_hot_label.shape)
        self.gradients[0] = self.outputs[-1].reshape(11,1) - one_hot_label.reshape(11,1)
        '''
        for layer_index in range(len(self.layers)):
            print(layer_index)
            print(1+layer_index)
            gradients[layer_index] = self.layers[-(1+layer_index)].backward(gradients[-(1+layer_index)])
        '''
        self.gradients[1] = self.layers[-1].backward(self.gradients[0])
        
        self.gradients[2] = self.layers[-2].backward(self.gradients[1])
        # Reshape delta_hidden1 for convolution    
        delta_conv_reshaped = self.gradients[2].reshape(62, 62, 1)
        
        # Correlate for gradient of convolution
        delta_kernel = cp.array(signal.correlate(
            img.reshape(64, 64, 3),
            delta_conv_reshaped,
            mode="valid"
        ))
        print(f"Gradients: {self.gradients[0].shape}")
        print(f"Gradients min/max: {cp.min(self.gradients[0])}, {cp.max(self.gradients[0])}")
        self.bias_convolved -= self.learn_rate * cp.sum(delta_conv_reshaped, axis=(0, 1))
        self.kernel_filter -= self.learn_rate * delta_kernel
    def plot_metrics(self, loss_history, accuracy_history):
        epochs = len(loss_history)
        
        # Plot loss
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(range(epochs), loss_history, color='blue', label='Loss')
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(range(epochs), accuracy_history, color='green', label='Accuracy')
        plt.title('Training Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        
        # Show both plots
        plt.tight_layout()
        plt.show()
    def train(self, epochs):
        images, labels = get_data()
        images = cp.asarray(images)
        labels = cp.asarray(labels)
        print("training...")
        total_loss = 0.0
        nr_correct = 0
        total_time = 0
        inc_learn_rate = .1
        n_incs = 0

        # Tracking lists for plotting
        loss_history = []
        accuracy_history = []

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
                img = img / 255.0  # Make sure it's between [0,1]
                img = (img - 0.5) / 0.5  # Normalize between [-1,1]
                output = self.forward(img)
                print(output) 
                nr_correct += int(cp.argmax(output) == cp.argmax(one_hot_label))
                
                # Cross-entropy loss for multi-class classification
                loss = -cp.sum(one_hot_label * cp.log(output + 1e-10))
                total_loss += loss
                
                # Call backward with proper arguments
                self.backward(img, output, one_hot_label)
            
            # Track loss and accuracy per epoch
            loss_history.append(total_loss / images.shape[0])
            accuracy_history.append((nr_correct / len(shuffled_indices)) * 100)

            # Show accuracy for this epoch
            end_time = time.time()
            delta_time = end_time - start_time
            nr_avg = round((nr_correct / len(shuffled_indices)) * 100, 2)
            avg_loss = total_loss / images.shape[0]
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Average Loss: {avg_loss.item():.4f}")
            print(f"Accuracy: {nr_avg}%")
            print(f"time taken: {delta_time}")
            total_time += delta_time

        print(f"total time was: {total_time} average time: {total_time/epochs}")
        
        # After training, plot the loss and accuracy
        self.plot_metrics(loss_history, accuracy_history)
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
    learn_rate = 0.1
    #path = r"D:\ai_projects\projects\data\training\Dessert\2.jpg"
    # Convert NumPy arrays to CuPy arrays

    neural_net = NeuralNetwork(n, learn_rate, size,3)
    if uinput == "train":
        epochs = 2
        neural_net.train(epochs)
        #neural_net.save()

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
