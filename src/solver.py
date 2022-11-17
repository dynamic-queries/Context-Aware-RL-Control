# %%
# Imports
import jax.numpy as jnp
from jax import jit
import numpy as np
import matplotlib.pyplot as plt
# %%
# Utils
def zero(A:np.ndarray):
    return np.zeros(A.shape)

class Constants:
    # Define universal constants
    def __init__(self):
        self.e = 1.6e-19 # C
        self.kb = 1.38e-34 # J/K
        self.me = 9.1e-31 # kg

class Containers:
    # Simulation fields 
    def __init__(self,Theta):
        self.Theta = Theta
        self.n = zero(Theta) # Density
        self.nn = zero(Theta) # Density of neutral atoms
        self.ui = zero(Theta) # Velocity of ion
        self.ue = zero(Theta) # Velocity of electron
        self.Te = zero(Theta) # Electron temperature
        self.Br = zero(Theta) # Radial magnetic field

class HETProblem:
    def __init__(self,constants:Constants,containers:Containers,tspan:tuple,tstep:float):
        nsteps = (tspan[1]-tspan[0])/tstep + 1
        self.tspan = tspan 
        self.tdomain = np.linspace(tspan[0],tspan[1],nsteps)
        self.constants = constants
        self.containers = np.array([containers for _ in range(len(self.tdomain))])

# %%
# Dependent functions - with closed form expressions

def Pe(Te,kb,ne):
    pass

def mu(mu0,omega):
    pass

def mu0(e,nue,me):
    pass

def omega(we,nue):
    pass

def nue(nua,nuei,nuc):
    pass

def nua():
    pass

def nuei():
    pass

def nuc():
    pass

def Br(x):
    pass

def ue(ue,j,e,n):
    pass

# %%
# Differential operators, Integral operators and their approximations

def neutral_con_equation():
    pass

def ion_electron_con_equation():
    pass

def ion_electron_mom_equation():
    pass

def electron_energy_equation():
    pass

def discharge_current():
    pass

# %%
# Initial condition routines


# %%
# simualtion routines

def solve(prob:HETProblem):
    pass
# %%
# Script
# Define the domain
Lc = 2.5 # cm
N = 1000 # no unit
Theta = np.linspace(0.0,Lc,N)
constants = Constants()
containers = Containers(Theta)
tspan = (0,1) # ms
tsave = 0.001
prob = HETProblem(constants,containers,tspan,tsave)
solution = solve(prob)