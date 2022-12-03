# %%
from jax import jit
import jax.numpy as np
import matplotlib.pyplot as plt
from derivatives import *
from solver import *
from utils import *

# %%
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

### Moral of the story (TODO) : Index forcing functions in solver.py accordingly. 

def test_solver_constructor():
    ## Define the space domain
    Lc = 2.5 # cm
    N = 1000 # no unit
    Theta = np.linspace(0.0,Lc,N)

    ## Define the time domain
    tspan = (0,1) # ms
    tsave = 0.001

    const = Constants()
    fields = Fields(Theta)
    params = Parameters(Theta,tspan)

    fields_arr = Array(fields)
    plot(fields_arr,Theta)
    plt.savefig("figures/solvers/ics.png")

    plt.clf()

    params_arr = Array(params)
    plot(params_arr,Theta)
    plt.savefig("figures/solvers/params.png")

    prob = HETProblem(const,fields,params,tspan,tsave)
    


# %% 
if __name__ == "__main__":
    test_first_derivative()
    plt.clf()
    test_second_derivative()
    test_solver_constructor()

# %%
