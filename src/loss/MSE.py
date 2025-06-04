import numpy as np
def MSE(y, y_pred):
    return (np.mean(y - y_pred))**2


def MSE_ET(y , y_pred):
    return (y - y_pred)


