import jax.numpy as np

def first_derivative(x):
    n = x.shape[0]
    dx = x[2]-x[1]
    d = -1*np.ones(n)
    d1 = np.ones(n-1)
    A = np.diag(d)
    A += np.diag(d1,k=1)
    A /= dx
    return A
    
def second_derivative(x):
    n = x.shape[0]
    dx = x[2]-x[1]
    d = 2*np.ones(n)
    d1 = -1*np.ones(n-1)
    A = np.diag(d) + np.diag(d1,k=1) + np.diag(d1,k=-1)
    A /= dx**2
    return A
