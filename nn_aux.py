import numpy as np

def sigmoid(z):
    """Sigmoid (logistic) function.

    z: a container of numbers
    Return: numpy array of sigmoid of each element of z
    """
    zz = np.array(z)
    return 1.0/(1.0+np.exp(-zz))


def sigmoid_grad(z):
    """Gradient of sigmoid (logistic) function.

    z: a container of numbers
    Return: numpy array of sigmoid_grad of each element of z
    """
    return sigmoid(z)*(1.0-sigmoid(z))


    
