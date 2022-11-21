# %%
from jax import jit
import jax.numpy as np
import matplotlib.pyplot as plt

# %%
from derivatives import *
def test_first_derivative():
    # Scripts for tests
    x = np.linspace(0.0,4*np.pi,100)
    y = np.sin(x)
    D = first_derivative(x)
    Dy = (D@y)[:-2]

    plt.plot(x,y)
    plt.plot(x[:-2],Dy)
    plt.xlabel("X")
    plt.ylabel("Y,DY")
    plt.title("First derivative test.")
    plt.legend(["sin(x)","cos(x)"])
    plt.savefig("figures/derivatives/first_deriv.png")

def test_second_derivative():
    x = np.linspace(0.0,4*np.pi,100)
    y = np.sin(x)
    D2 = second_derivative(x)
    D2y = (D2@y)[1:-2]

    plt.plot(x,y)
    plt.plot(x[1:-2],D2y)
    plt.xlabel("X")
    plt.ylabel("Y,D2Y")
    plt.title("Second derivative test.")
    plt.legend(["sin(x)","~sin(x)"])
    plt.savefig("figures/derivatives/second_deriv.png")


if __name__ == "__main__":
    test_first_derivative()
    plt.clf()
    test_second_derivative()

### Moral of the story (TODO) : Index forcing functions in solver.py accordingly. 