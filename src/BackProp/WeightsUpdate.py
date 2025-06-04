import numpy as np
def NewWeightValue(weight_old, gradient_value, lr: int = 0.1):
    """
    Update the weight value using the gradient and learning rate.

    Parameters:
        weight_old (np.ndarray): The current weight values.
        gradient_value (np.ndarray): The gradient of the loss with respect to the weights.
        lr (float, optional): The learning rate. Default is 0.1.

    Returns:
        np.ndarray: The updated weight values.
    """
    weight_new = np.zeros_like(weight_old)
    weight_new = weight_old - lr * gradient_value
    return weight_new


def NewBiasValue(bias_old, bias_grad, lr=0.1):
    """
    Update the bias value using the gradient and learning rate.

    Parameters:
        bias_old (np.ndarray or float): The current bias values.
        bias_grad (np.ndarray or float): The gradient of the loss with respect to the bias.
        lr (float, optional): The learning rate. Default is 0.1.

    Returns:
        np.ndarray or float: The updated bias values.
    """
    bias_new = bias_old - lr * bias_grad
    return bias_new