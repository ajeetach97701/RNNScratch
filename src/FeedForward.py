import numpy as np

"""
Let us consider that there is only one hidden layer for now.
input -> hidden -> output

"""
def feed_forward(num_neurons:int, data):
    prev_hidden = None
    data_array = data.to_numpy()
    outputs = []

    _, y  = data_array.reshape(-1,1).shape
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
    o_weights = np.random.rand(y, num_neurons)
    
    h_bias = np.random.rand(3,1) # assume the oupout layer size is 1


    o_bias = np.random.rand(1,1)

    
    
    
    prev_hidden = np.random.rand(num_neurons, 1)
    print(prev_hidden.shape)
        
    for i in data_array:
        input_data = i.reshape(-1,1)


        hidden_raw = i_weights @ input_data + h_weights @ prev_hidden + h_bias
        
        prev_hidden = np.tanh(hidden_raw)
        output = o_weights @ hidden_raw + o_bias
        outputs.append(output)
    return outputs