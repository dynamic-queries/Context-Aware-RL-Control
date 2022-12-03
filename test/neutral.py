# %% 
import jax.numpy as np
from jax import jit
import matplotlib.pyplot as plt
import diffrax as diffrax
import pandas as pd
import equinox as eqx

# %% Utils
def first_derivative(x):
    n = x.shape[0]
    dx = x[2]-x[1]
    d = -1*np.ones(n)
    d1 = np.ones(n-1)
    A = np.diag(d)
    A += np.diag(d1,k=1)
    A /= dx
    return A


# %% 
class Constants:
    def __init__(self):
        self.e = 1.6e-19 # C
        self.mi = 1.3284e-29 # kg 
        self.kb = 1.38e-23 # J/K

class Fields:

    def __init__(self,Z):
        self.Z = Z
        n = Z.shape[0]

        self.rho_n = np.zeros(n)
        self.rho_p = np.zeros(n)
        self.u_n = np.ones(n)
        self.u_p = np.zeros(n)
        self.u_e = np.zeros(n)
        self.P_i = np.zeros(n)
        self.P_e = np.zeros(n)
        self.Te = np.zeros(n)
        self.eval()

    def eval(self):
        """Evaluate initial guess conditions.
        """
        self.rho_N() 
        self.rho_P()
        self.u_P()
        self.u_E()
        self.T_E()

    def rho_N(self):
        Z = self.Z
        sigma = 0.5
        xmean = 0.0
        self.rho_n = 12e18*np.exp(-((Z-xmean)/sigma)**2)

    def rho_P(self):
        Z = self.Z
        xmean = 0.5
        sigma = 0.3
        self.rho_p =  6e17*np.exp(-((Z-xmean)/sigma)**2)
    
    def u_P(self):
        Z = self.Z
        self.u_n =  2e5*(1/(1+np.exp(-2*Z)))

    def u_E(self):
        Z = self.Z
        self.u_e = -3+(1/(1+np.exp(-1.5*Z)))

    def T_E(self):
        Z = self.Z
        mean = 0.75
        sigma = 1.0
        self.Te = 30*np.exp(-((Z-mean)/sigma)**2)

class Params:

    def __init__(self,Z):
        n = Z.shape[0]
        self.kI = np.zeros(n)
        self.nW = np.zeros(n)
        self.K = np.zeros(n)
        self.W = np.zeros(n)
        self.mu = np.ones(n)

    def ionization_coefficient(self,Y):
        ionization = pd.read_csv("ionization_data.dat",sep="\t")
        ionization = np.transpose(ionization.to_numpy())
        E = ionization[0,:]
        ki = ionization[1,:]
        print(E,ki)
        self.kI = np.interp(Y,E,ki)

    def reconstruction_rate(self,te,mi,kb,r1,r2,ni,alpha):
        vwi = alpha*np.sqrt((kb*te)/(mi)) * (2/(r2-r1))
        self.nW =  ni*vwi

# %% Governing equations
def neutral_continuity(fields,parameters,constants):
    rho_n = fields.rho_n
    rho_p = fields.rho_p
    u_n = fields.u_n
    kI = parameters.kI
    nW = parameters.nW
    
    return -D@(rho_n*u_n) - kI*rho_n*rho_p + nW

def plasma_momentum_continuity(fields,parameters,constants):
    rho_p = fields.rho_p
    u_p = fields.u_p
    rho_n = fields.rho_n
    u_e = fields.u_e
    P_i = fields.P_i
    P_e = fields.P_e

    kI = parameters.kI
    nW = parameters.nW
    mu = parameters.mu

    e = constants.e
    mi = constants.mi

    v1 = -rho_p * (D@(rho_p*u_p)) + (rho_p*rho_p*rho_n*kI) - rho_p*nW
    v2 = -u_p*(D@(rho_p*u_p)) - (u_p*rho_p*rho_n*kI) + u_p*nW - (e*rho_p*u_e)/(mi*mu) - D@(rho_p*u_p*u_p + (P_i+P_e)/mi)
    return np.vstack((v1,v2)) 

def electron_temperature(fields,parameters,constants):
    rho_p = fields.rho_p
    rho_n = fields.rho_n
    u_e = fields.u_e
    Te = fields.Te
    P_e = fields.P_e

    e = constants.e
    kb = constants.kb
    
    K = parameters.K
    W = parameters.W
    mu = parameters.mu
    
    return -D@(5*rho_p*kb*Te*u_e/2) + D@(5*mu*rho_p*kb*kb*Te*(D@Te))/2 + u_e*D@(P_e) + (rho_p*u_e*u_e*e)/mu - rho_n*rho_p*K - rho_p*W 

def integrate():
    pass 

# %% 
@jit
def solver():
    Nz = 100
    Zf = 2.5
    Z = np.linspace(0.0,Zf,Nz)

    D = first_derivative(Z)
    fields = Fields(Z)
    constants = Constants()

    # Constants and intermediate variables for computing the parameters
    alpha = 0.115
    E = 300
    Y = E/fields.rho_n
    R1 = 210e-3
    R2 = 90e-3
    ni = fields.rho_p
    te = fields.Te
    kb = constants.kb
    mi = constants.mi

    parameters = Params(Z)
    parameters.ionization_coefficient(Y);
    parameters.reconstruction_rate(te,mi,kb,R1,R2,ni,alpha);

    # Setup ODE Problem
    t0 = 0.0
    t1 = 1e-3
    dt0 = 1e-6
    y0 = fields.rho_n
    y0
    args = [fields.rho_p,fields.u_n,parameters.kI,parameters.nW]

    def forcing_function(t,y,args):
        rho_p,u_n,kI,nW = args
        return -D@(y*u_n) - kI*y*rho_p + nW

    term = diffrax.ODETerm(forcing_function)
    solver = diffrax.Heun()
    solution = diffrax.diffeqsolve(term, solver,t0, t1, dt0, y0,args=args)
    return y0,solution