
from src.BackProp import GradHidden, GradType, NewWeightValue, NewBiasValue
from src.loss import MSE, MSE_ET


def backProp(hidden_input, weight_hidden, weight_input, weight_output, y_hat, y, x_input, hidden_input_prev, ht):
    """
    Perform a single step of backpropagation through time (BPTT) for an RNN.

    This function calculates the gradients of the loss with respect to the output, hidden,
    and input weights, using the current and previous hidden states, predicted and target outputs,
    and the current input. It then updates the weights accordingly.

    Parameters:
        hidden_input (np.ndarray): The hidden state at current time step t (h_t), shape (hidden_size, 1).
        weight_hidden (np.ndarray): The weight matrix from hidden to hidden (W_h), shape (hidden_size, hidden_size).
        weight_input (np.ndarray): The weight matrix from input to hidden (W_i), shape (hidden_size, input_size).
        weight_output (np.ndarray): The weight matrix from hidden to output (W_o), shape (output_size, hidden_size).
        y_hat (np.ndarray): Predicted output at time t, shape (output_size, 1).
        y (np.ndarray): True target output at time t, shape (output_size, 1).
        x_input (np.ndarray): Input at time t, shape (input_size, 1).
        hidden_input_prev (np.ndarray): Hidden state from previous time step t-1 (h_{t-1}), shape (hidden_size, 1).
        ht (np.ndarray): Hidden state at current time step (used for output gradient), shape (hidden_size, 1).

    Returns:
        tuple: Updated weight matrices (weight_input, weight_hidden, weight_output).
    """
    #  For the output weight updating
    et = MSE_ET(y = y , y_pred= y_hat)

    output_gradient_value = GradHidden(
        et= 0,
        weight = weight_output,
        ht = ht,
        type = GradType.OUTPUT,
        )
    weight_output = NewWeightValue(weight_old=weight_output, gradient_value=output_gradient_value)
    
    hidden_gradient_value = GradHidden(
        et= et, # Loss value
        weight = weight_hidden, # Hidden weight value
        ht = hidden_input, # ht for calculating 1-ht^2
        type = GradType.HIDDEN,  #type checking
        x = hidden_input_prev, # For gradient calculation  δ_t @ ht-1.T
        )
    weight_hidden = NewWeightValue(weight_old=weight_hidden, gradient_value=hidden_gradient_value)
    input_gradient_value = GradHidden(
        et= et,
        weight = weight_input,
        ht = hidden_input,
        type = GradType.INPUT,
        x = x_input
        )
    weight_input = NewWeightValue(weight_old=weight_input, gradient_value=input_gradient_value)
    return weight_input, weight_hidden, weight_output