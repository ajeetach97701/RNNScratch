
import numpy as np
from enum import Enum


class GradType(Enum):
    OUTPUT = "output"
    HIDDEN = "hidden"
    INPUT = "input"
    

def GradHidden(et, weight, ht, type:GradType, x = None):
    """
    Compute the gradient of the hidden layer weights in a recurrent neural network (RNN) during backpropagation.

    This function calculates the gradient for the hidden layer weight matrices by backpropagating the error
    from the output layer through the hidden layer's nonlinearity and combining it with the appropriate input.

    Mathematical Background:
        In RNN backpropagation through time (BPTT), the gradient of the hidden layer weights is computed as:
            δ_t = (W_o.T @ e_t) ⊙ (1 - h_t^2)
            grad = δ_t @ x.T
        where:
            - e_t: error at the output layer at time t, shape (output_size, 1)
            - W_o: weights from hidden to output, shape (output_size, hidden_size)
            - h_t: hidden state at time t, shape (hidden_size, 1)
            - x: input to the weight matrix being updated (see below for details)
        The operation ⊙ denotes element-wise multiplication.

    Parameter Shapes:
        et (np.ndarray): Error term from the output layer at time t.
            Shape: (output_size, 1)
        weight (np.ndarray): Weight matrix from hidden to output layer.
            Shape: (output_size, hidden_size)
        ht (np.ndarray): Hidden layer activations at time t.
            Shape: (hidden_size, 1)
        x (np.ndarray): Input for gradient calculation:
            - If computing gradient for input weights (W_xh): x is the input vector at time t, shape (input_size, 1)
            - If computing gradient for hidden weights (W_hh): x is the hidden state from previous time step, shape (hidden_size, 1)

    Returns:
        np.ndarray: Gradient of the hidden layer weights.
            - Shape: (hidden_size, input_size) for input weights, or (hidden_size, hidden_size) for hidden weights,
            matching the shape of the weight matrix being updated.

    Notes:
        - The function assumes the use of tanh nonlinearity for the hidden layer (hence the (1 - h_t^2) derivative).
        - For correct broadcasting, ensure that x is a column vector (2D array of shape (input_size, 1) or (hidden_size, 1)).
    """
    if type == GradType.HIDDEN or type == GradType.INPUT:
        if x is None:
            raise Exception("NoneType not allowed for GradType.INPUT or GradType.HIDDEN")
        del_t = np.transpose(weight) @ et  * (1- ht**2)
        gradient = del_t @ x
        return gradient
    elif GradType.OUTPUT:
        gradient = et * ht
        return gradient
    else:
        raise Exception("Type not found")
        
        