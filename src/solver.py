# %%
# Imports
from derivatives import *
from utils import *

import jax.numpy as np
from jax import jit
import matplotlib.pyplot as plt


# %%  
# Types

class Constants:
    # Define universal constants
    def __init__(self):
        self.e = 1.6e-19 # C
        self.kb = 1.38e-34 # J/K
        self.me = 9.1e-31 # kg
        self.mi = 9.1e-29 # kg

class Fields:
    # Simulation fields 
    def __init__(self,Theta):
        self.Theta = Theta
        self.n = zero(Theta) # Density
        self.nn = zero(Theta) # Density of neutral atoms
        self.ui = zero(Theta) # Velocity of ion
        self.ue = zero(Theta) # Velocity of electron
        self.Te = zero(Theta) # Electron temperature
        self.Pe = zero(Theta)
        self.Pi = zero(Theta)
        self.update_ic() # Update initial conditions

    def update_ic(self):
        pass


class Parameters:
    # Simulation parameters
    def __init__(self,Theta,tspan):
        self.Br = zero(Theta)
        self.mflow = 0.0
        self.kI = zero(Theta)
        self.nw = zero(Theta)
        self.K = zero(Theta)
        self.W = zero(Theta)
        self.update()

    def update(self):
        pass


class HETProblem:
    def __init__(self,constants:Constants,containers:Fields,params:Parameters,tspan:tuple,tstep:float):
        # 
        nsteps = (tspan[1]-tspan[0])/tstep + 1
        self.tspan = tspan 
        self.tdomain = np.linspace(tspan[0],tspan[1],nsteps)
        self.constants = constants
        self.containers = np.array([containers for _ in range(len(self.tdomain))])
        self.parameters = params

        # Precompute the derivative matrices
        self.Dx = first_derivative(containers.Theta)
        self.Dxx = second_derivative(containers.Theta)

# %%
# Dependent functions - simple closed form expressions

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
# Forcing functions for the ODEs

def neutral_con_equation(prob:HETProblem,k:int):
    # Get fields
    nn = prob.fields[k-1].nn
    n = prob.fields[k-1].n
    un = prob.fields[k-1].un
    #
    kI = prob.parameters.kI
    nw = prob.parameters.nw
    #
    Dx = prob.Dx
    
    return -Dx@ nn*un - nn*n*kI + nw

def ion_electron_con_equation(prob:HETProblem,k:int):
    #
    n = prob.fields[k-1].n
    ui = prob.fields[k-1].ui
    nn = prob.fields[k-1].nn
    #
    kI = prob.parameters.kI
    nw = prob.parameters.nw
    #
    Dx = prob.Dx

    return -Dx@ n*ui + kI*nn*n - nw
    
def ion_electron_mom_equation(prob:HETProblem,k:int):
    #
    n = prob.fields[k-1].n
    ui = prob.fields[k-1].ui
    pi = prob.fields[k-1].pi
    pe = prob.fields[k-1].pe
    #
    mu = prob.parameters.mu
    #
    e = prob.constants.e
    mi = prob.constants.mi
    Dx = prob.Dx

    temp = n*ui**2 + (pe + pi)/mi
    return -Dx@temp - (n*ui*e)/(mi*mu)

def electron_energy_equation(prob:HETProblem,k:int):
    # 
    n = prob.fields[k-1].n
    nn = prob.fields[k-1].n
    pe = prob.fields[k-1].pe
    Te = prob.fields[k-1].Te
    ue = prob.fields[k-1].ue
    #
    K = prob.parameters.K
    W = prob.parameters.W
    mu = prob.parameters.mu

    #
    kb = prob.constants.kb
    e = prob.constants.e
    Dx = prob.Dx

    temp1 = (5*kb*e*Te*ue)/(2)
    temp2 = (5*kb**2 *e*Te*mu)/(e)
    return -Dx @ temp1 + Dx @ (temp2*(Dx@Te)) + ue*(Dx@pe) + n*ue**2 *(e/mu) - n*nn*K - n*W
    

def discharge_current():
    # I need a differentiable package that does Gaussian Quadrature
    pass

# %%
# Initialize initial conditions and other parameters of the model



# %%
# Simualtion routines


def bc(containers,precomputed_bc):
    pass

def integrate(prob,gradients,k):
    """
    Integrate is written to not update the cells with the boundary conditions
    """
    pass

def update(J,ue,ts):
    pass 

def solve(prob:HETProblem):
    # Get parameters.
    tdomain = prob.tdomain

    # Time step loop.
    for ts,t in enumerate(1,tdomain):
        f_nn = neutral_con_equation(prob,ts)
        f_niui = ion_electron_mom_equation()
        f_ni = ion_electron_con_equation()
        f_Te = electron_energy_equation()
        gradients = [f_nn,f_niui,f_ni,f_Te]
        integrate(prob,gradients,ts)
        J = discharge_current()
        ue_ = ue()
        update(J,ue_,ts)
