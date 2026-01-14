import numpy as np

def normalize_edges(E):
    """Normalizes the edge feature matrix according to https://arxiv.org/pdf/1809.02709

    Args:
        E (ndarray): 3D numpy array for the edge feature matrix with shape N x N x F

    Returns:
        ndarray: normalized edge feature matrix with shape N x N x F
    """
    E_hat = np.zeros_like(E)
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            for p in range(E.shape[2]):
                E_hat[i,j,p] = E[i,j,p]/np.sum(E[i,:,p])

    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            for p in range(E.shape[2]):
                sum = 0
                for k in range(E.shape[1]):
                    sum += E_hat[i,k,p]*E_hat[j,k,p]/np.sum(E_hat[:,k,p])
                E[i,j,p] = sum
    return E