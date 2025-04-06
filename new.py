import numpy as np

def one_hot_encode(values, n_classes=None):
    """
    One-hot encodes categorical values using NumPy.
    Works with column vectors of shape (n, 1).
    
    Parameters:
    values (array-like): Column vector of shape (n, 1) with categorical values
    n_classes (int, optional): Number of distinct categories. If None, inferred from data.
    
    Returns:
    numpy.ndarray: 2D array where each row corresponds to a one-hot encoded vector
    """
    # Handle (n, 1) shaped arrays by flattening first
    values = np.asarray(values).flatten()
    
    if n_classes is None:
        n_classes = np.max(values) + 1
    
    # Create a zero matrix of shape (n_samples, n_classes)
    one_hot = np.zeros((values.size, n_classes))
    
    # Set the appropriate indices to ones
    one_hot[np.arange(values.size), values] = 1
    
    return one_hot

# Example with shape (11,1)
if __name__ == "__main__":
    # Sample data with shape (11,1)
    categories = np.array([[0], [1], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
    print("Original shape:", categories.shape)
    
    # Perform one-hot encoding
    encoded = one_hot_encode(categories)
    
    print("Original categories:", categories.flatten())
    print("One-hot encoded shape:", encoded.shape)
    print("One-hot encoded:\n", encoded)
