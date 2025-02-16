import numpy as np
from data import get_data
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from scipy import signal
from scipy import special
from resize import ResizeIMG_64_64
from PIL import Image
#math funcitons
#activation funcitons
def fast_gelu(x):
    return 1 / (1 + np.exp(-x))

def delta_fast_gelu(x):
    return (x * (1 - x))

class NeuralNetwork:
    def __init__(self, n, learn_rate, size, kernel_size):
        #set hyper parameters
        self.array = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat", "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit"]
        self.n = n
        self.learn_rate = learn_rate
        self.kernel_filter = np.random.uniform(-0.5,0.5, (kernel_size,kernel_size,kernel_size))
        #set weights
        entrance_size = (size-kernel_size + 1)**2
        #                                                        (62,62)
        self.weight_input_hidden1 = np.random.uniform(-0.5, 0.5, (11532,))
        self.weight_hidden1_hidden2 = np.random.uniform(-0.5, 0.5, (11532,))
        self.weight_hidden2_hidden3 = np.random.uniform(-0.5, 0.5, (11532, ))
        self.weight_hidden3_output = np.random.uniform(-0.5, 0.5, (11,11532))
        print(self.weight_hidden3_output.shape)
        #set biases
        self.bias_input_hidden1 = np.zeros((11532, ))
        self.bias_hidden1_hidden2 = np.zeros((11532, ))
        self.bias_hidden2_hidden3 = np.zeros((11532, ))
        self.bias_hidden3_output = np.zeros((11, ))
        self.bias_convolved = np.zeros((62,62,3))
        
    def forward(self, x):
        print("here")
        print(x.shape)
        print(self.kernel_filter.shape)
        self.convolved_unbias = signal.fftconvolve(x,self.kernel_filter,mode="valid")
        self.convolved = self.convolved_unbias + self.bias_convolved
        print(self.convolved.shape)
        self.convolved = self.convolved.reshape(11532,)
        # Forward propagation input -> hidden1
        hidden1_pre = self.bias_input_hidden1 + (self.weight_input_hidden1 @ self.convolved)
        self.hidden1 = fast_gelu(hidden1_pre)
        print(self.hidden1.shape, "HIDDEN SHAPE")
        # Forward propagation hidden1 -> hidden2
        hidden2_pre = self.bias_hidden1_hidden2 + (self.weight_hidden1_hidden2 @ self.hidden1)
        self.hidden2 = fast_gelu(hidden2_pre)

        # Forward propagation hidden2 -> hidden3
        print(self.weight_hidden2_hidden3.shape, self.hidden2.shape)
        hidden3_pre = self.bias_hidden2_hidden3 + (self.weight_hidden2_hidden3 @ self.hidden2)
        self.hidden3 = fast_gelu(hidden3_pre)
        print(self.weight_hidden3_output.shape, self.hidden3.shape,"GRRRRRRRRRRRR") 
        # Forward propagation hidden2 -> output
        output_pre = self.bias_hidden3_output + (self.weight_hidden3_output @ self.hidden3)
        self.output = special.softmax(output_pre)
        print("HERHERKJFHSDAKFJHSADFHSADLKF2")
        print(self.output.shape)
        return self.output
    def backward(self,img,output,label):
            # Backpropagation output -> hidden3 (cost function derivative)
            delta_output = output - label.reshape(11,)
            print(delta_output.shape, output.shape,self.output.shape)
            print("BACK PROP")
            print(self.weight_hidden3_output.shape,delta_output.shape, self.output.T.shape)
            self.weight_hidden3_output -= self.learn_rate * (delta_output @ self.output.T)
            self.bias_hidden3_output -= self.learn_rate * delta_output

            # Backpropagation hidden3 -> hidden2 (activation function derivative)
            delta_hidden3 = self.weight_hidden3_output.T @ delta_output * delta_fast_gelu(self.hidden3)
            self.weight_hidden2_hidden3 -= self.learn_rate * delta_hidden3 @ self.hidden3.T
            self.bias_hidden2_hidden3 -= self.learn_rate * delta_hidden3
            #
            #MIGHT NOT NEED TO TRANSPOSE WIGHTS FOR DELTA HIDDEN
            #
            # Backpropagation hidden2 -> hidden1 (activation function derivative)
            delta_hidden2 = self.weight_hidden2_hidden3 @ delta_hidden3 * delta_fast_gelu(self.hidden2)
            self.weight_hidden1_hidden2 -= self.learn_rate * delta_hidden2 @ self.hidden2.T
            self.bias_hidden1_hidden2 -= self.learn_rate * delta_hidden2

            # Backpropagation hidden1 -> self.convolved (activation function derivative)
            delta_hidden1 = self.weight_hidden1_hidden2 @ delta_hidden2 * delta_fast_gelu(self.hidden1)
            print(delta_hidden1)
            print("ASDASDASDASD")
            self.weight_input_hidden1 -= self.learn_rate * delta_hidden1 @ self.hidden1.T
            self.bias_input_hidden1 -= self.learn_rate * delta_hidden1

            #Backpropagation convoled -> input
            print(delta_hidden1.shape)
            k_gradient = np.zeros(self.kernel_filter.shape)
            k_gradient = signal.fftconvolve(img.reshape(64,64,3),delta_hidden1.T.reshape(62,62,3), "valid")
            #bias_gradient = signal.fftconvolve(img.reshape(64,64,3),self.kernel_filter, "valid")
            self.bias_convolved -= self.learn_rate*(self.convolved-delta_hidden1).reshape(62,62,3)
            self.kernel_filter -= self.learn_rate * k_gradient
            print("END OF BACK PROP")
    def train(self, epochs):
        images, labels = get_data()
        images = np.asarray(images)
        labels = np.asarray(labels)
        print("training...")
        total_loss = 0.0
        nr_correct = 0
        total_time = 0

        for epoch in range(epochs):
            shuffled_indices = np.random.permutation(images.shape[0])
            start_time = time.time()
            for idx in shuffled_indices:
                print("NEW GEN")
                img = images[idx]
                label = labels[idx]
                img.shape += (1,)
                label.shape += (1,)
                one_hot_label = np.zeros((11,))
                one_hot_label[label] = 1
                #brute force change for testing purpouses
                img1 = np.sum(ResizeIMG_64_64("./bread.jpg"),axis=2)
                #label = np.asarray([1,0,0,0,0,0,0,0,0,0,0])
                print(img.shape,"IMG SHAPE") 
                print(img1.shape, "IDEAL SHAPE")
                img = img.reshape(64,64,3)
                output = self.forward(img)
                nr_correct += int(np.argmax(output) == np.argmax(label))
                print(one_hot_label.shape, output.shape)
                loss = -one_hot_label * np.log(output) + (1 - one_hot_label) * np.log(1 - output)
                total_loss += loss
            
                self.backward(img,output,label)
            print(len(shuffled_indices))
            # Show accuracy for this epoch
            end_time = time.time()
            delta_time = end_time - start_time
            nr_avg = round((nr_correct / images.shape[0]) * 100, 2)
            avg_loss = total_loss / images.shape[0]
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Average Loss: {np.mean(avg_loss).item():.4f}")
            print(f"Accuracy: {nr_avg}%")
            print(f"time taken: {delta_time}")
            total_time += delta_time
            nr_correct = 0
           # if nr_avg >= 40:
               # self.learn_rate *=.1
        print(f"total time was: {total_time} average time: {total_time/epochs}")
    def save(self):
        np.savez("neuralNetworkData.npz", 
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
        data = np.load("neuralNetworkData.npz")
        
        self.weight_input_hidden1 = np.asarray(data["weight_input_hidden1"])
        self.weight_hidden1_hidden2 = np.asarray(data["weight_hidden1_hidden2"])
        self.weight_hidden2_hidden3 = np.asarray(data["weight_hidden2_hidden3"])
        self.weight_hidden3_output = np.asarray(data["weight_hidden3_output"])
        
        self.bias_input_hidden1 = np.asarray(data["bias_input_hidden1"])
        self.bias_hidden1_hidden2 = np.asarray(data["bias_hidden1_hidden2"])
        self.bias_hidden2_hidden3 = np.asarray(data["bias_hidden2_hidden3"])
        self.bias_hidden3_output = np.asarray(data["bias_hidden3_output"])
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

        plt.title(f"Is it {neural_net.array[np.argmax(output).item()]} :)")
        print(np.exp(output)/np.sum(np.exp(output)))
        print(np.argmax(output))
        plt.show()

if __name__ == "__main__":
    main()
