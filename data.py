import numpy as np
import random
def get_data():
    print("getting data...")
    with np.load("dataSample.npz") as f:
        images, labels = f["data"], f["labels"]
    temp = list(zip(images, labels))
    random.shuffle(temp)
    images, labels = zip(*temp)
    images, labels = np.array(images), np.array(labels)
    images = images.astype("float32") / 255
    images = np.reshape(images, (images.shape[0], -1))  # Reshape to a flat vector
    return images, labels
# Print the shapes of the loaded data (optional)
