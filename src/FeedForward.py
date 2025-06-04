import numpy as np

"""
Let us consider that there is only one hidden layer for now.
input -> hidden -> output

"""
def feedForward(num_neurons:int, out_neurons:int, data):
    """
Performs a feed-forward pass through a neural network with one hidden layer.

Parameters:
    num_neurons (int): The number of neurons in the hidden layer.
    out_neurons (neurons): Number of neurons in the output layer
    data: The input data, expected to be convertible to a NumPy array.

Returns:
    list: The output values from the network after processing the input data.
"""
    
    
    
    prev_hidden = None
    data_array = data.to_numpy()
    
    hidden_values = []
    outputs = []

    _, y  = data_array.reshape(-1,1).shape
    print("------------------------",data_array)
    
    
    
    
    """
    Weight initialization:
        We need the following weights:
            1. input weight
            2. Hidden layer weight
            3. Output layer weight
    Bias initialization:
    We need the following biases:
        1. hidden layer bias
        2. Output layer bias
    """
    i_weights = np.random.rand(num_neurons, y)


    
    h_weights = np.random.rand(num_neurons, num_neurons)
    o_weights = np.random.rand(out_neurons, num_neurons)
    
    h_bias = np.random.rand(num_neurons,1) # assume the oupout layer size is 1


    o_bias = np.random.rand(out_neurons,1)

    
    
    
    prev_hidden = np.random.rand(num_neurons, 1)
    print(prev_hidden.shape)
        
    for i in data_array:
        input_data = i.reshape(-1,1)


        hidden_raw = i_weights @ input_data + h_weights @ prev_hidden + h_bias
        hidden_values.append(hidden_raw)
        
        prev_hidden = np.tanh(hidden_raw)
        output = o_weights @ prev_hidden + o_bias
        outputs.append(output)
    return outputs