from scipy.constants import Boltzmann
import numpy as np
import os
if 'READTHEDOCS' not in os.environ:
    from solve import solve as solve_cython
from frange import frange

class sde_solver():
    """
    Solves the following SDE for q(t) and d(q(t))/d(t):
    d^2(q(t))/d(t)^2 = -[Γ0 - Ω0 η q(t)^2]*d(q(t))/d(t) - Ω0^2*q(t)^2 + sqrt(2*Γ0*kB*T0/m)*d(W(t))/d(t)
    
    Using the Euler-Maruyama method.

    Where:
    q(t) is the position of the particle (x, y or z) with time
    Ω0 is the trapping frequency
    Γ0 is the damping due to the environment
    kB is the Boltzmann constant
    T0 is the environment temperature
    m is the mass of the nanoparticle
    η is the modulation depth of the cooling signal ???
    W(t) is the Wiener process
    """
    def __init__(self, Omega0, Gamma0, deltaGamma, mass, T0=300, q0=0, v0=0, TimeTuple=[0, 100e-6], dt=1e-9, seed=None):
        """
        Initialises the sde_solver instance.

        Parameters
        ----------
        Omega0 : float
            Trapping frequency
        Gamma0 : float
            Enviromental damping
        deltaGamma : float
            damping due to other effects (e.g. feedback cooling)
        mass : float
            mass of nanoparticle (in Kg)
        T0 : float, optional
            Temperature of the environment, defaults to 300
        q0 : float, optional
            initial position, defaults to 0
        v0 : float, optional
            intial velocity, defaults to 0
        TimeTuple : tuple, optional
            tuple of start and stop time for simulation / solver
        dt : float, optional
            time interval for simulation / solver
        seed : float, optional
            random seed for generate_weiner_path, defaults to None
            i.e. no seeding of random numbers
        
        """
        self.k_B = Boltzmann # J/K
        self.tArray = np.arange(0, 500e-6, dt)
        self.q0 = q0
        self.v0 = v0
        self.Omega0 = Omega0
        self.Gamma0 = Gamma0
        self.deltaGamma = deltaGamma
        self.mass = mass
        self.T0 = T0
        self.TimeTuple = TimeTuple
        self.b_v = np.sqrt(2*self.Gamma0*self.k_B*self.T0/self.mass) # a constant
        self.dt = dt
        self.tArray = frange(TimeTuple[0], TimeTuple[1], dt)
        self.generate_weiner_path(seed)
        
        self.q = np.zeros(len(self.tArray)) # initialises position array, q
        self.v = np.zeros(len(self.tArray)) # initialises velocity array, v
        self.q[0] = self.q0 # sets initial position to q0
        self.v[0] = self.v0 # sets initial position to v0
        self.SqueezingPulseArray = np.ones(len(self.tArray)) # initialises squeezing pulse array such that there is no squeezing
        return None

    def generate_weiner_path(self, seed=None):
        """
        Generates random values of dW along path and populates
        self.dwArray property with values along with path.
        The values of dW are independent and identically 
        distributed normal random variables with expected 
        value 0 and variance dt. Sets the dwArray property.

        seed : float, optional
            random seed for generate_weiner_path, defaults to None
            i.e. no seeding of random numbers

        """
        if seed != None:
            np.random.seed(seed)
        self.dwArray = np.random.normal(0, np.sqrt(self.dt), len(self.tArray))
        return None

#    def a_q(self, t, p, q): # replaced with just p to reduce slow python function evaluations
#        return p

    def a_v(self, q, v):
        return _a_v(q, v, self.Gamma0, self.Omega0, self.eta)
        
    def solve(self, NumTimeSteps=None, startIndex=0):
        """
        Solves the SDE from timeTuple[0] to timeTuple[1]
        
        Parameters
        ----------
        NumTimeSteps : int, optional
            number of time steps to solve for
        startIndex : int, optional
            array index (of q and v) at which to start solving the SDE

        Returns
        -------
        self.q : ndarray
            array of positions with time
        self.v : ndarray
            array of velocities with time
        """
        if NumTimeSteps == None:
            NumTimeSteps = (len(self.tArray) - 1) - startIndex
        self.q, self.v = solve_cython(self.q, self.v, float(self.dt), self.dwArray, float(self.Gamma0),float(self.deltaGamma), float(self.Omega0), float(self.b_v), SqueezingPulseArray=self.SqueezingPulseArray, startIndex=startIndex, NumTimeSteps=NumTimeSteps)
        return self.q, self.v

#def _a_v(q, v, Gamma0, Omega0, eta):
#    return -(Gamma0 + deltaGamma)*v - Omega0**2*q

