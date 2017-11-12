from scipy.constants import Boltzmann
import numpy as np
import os
if 'READTHEDOCS' not in os.environ:
    from optosim.solveRK_kalman import solve as solver
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
    def __init__(self, Omega0, Gamma0, deltaGamma, mass,
                 T0=300, q0=0, v0=0, alpha=0, beta=0,
                 DoubleFreqAmplitude=0, DoubleFreqPhaseDelay=0,
                 SingleFreqAmplitude=0, SingleFreqPhaseDelay=0,
                 liadTau=0,                 
                 TimeTuple=[0, 100e-6], dt=1e-9,
                 TimeAfterWhichToApplyFeedback=0,
                 seed=None,
                 filter_sample_freq=None,
                 x_initial=None,         
                 P_initial=None,         
                 F=None,                 
                 H=None,                 
                 Q=None,                 
                 R=None,
                 KalmanCoolingAmplitude=None,
                 KalmanCoolingTimeDelay=None,
                 KalmanAmplitudeControlWindow=None):
        """
        Initialises the sde_solver instance.

        Parameters
        ----------
        Omega0 : float
            Trapping frequency
        Gamma0 : float
            Enviromental damping - in radians/s - appears as (-Gamma*v) term in the SDE
        deltaGamma : float
            damping due to other effects (e.g. feedback cooling) - in radians/s - appears as (-deltaGamma*q**2*v)*dt term in the SDE
        mass : float
            mass of nanoparticle (in Kg)
        T0 : float, optional
            Temperature of the environment, defaults to 300
        q0 : float, optional
            initial position, defaults to 0
        v0 : float, optional
            intial velocity, defaults to 0
        alpha : float
            prefactor multiplying the q**3 non-linearity term shows up as ([alpha*q]**3*dt) in the SDE
        beta : float
            prefactor multiplying the q**5 non-linearity term shows up as ([beta*q]**5*dt) in the SDE
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
        self.alpha = alpha
        self.beta = beta
        self.DoubleFreqAmplitude = DoubleFreqAmplitude
        self.DoubleFreqPhaseDelay = DoubleFreqPhaseDelay
        self.SingleFreqAmplitude = SingleFreqAmplitude
        self.SingleFreqPhaseDelay = SingleFreqPhaseDelay
        self.liadTau = liadTau

        self.filter_sample_freq = filter_sample_freq
        self.x_initial =          x_initial
        self.P_initial =          P_initial
        self.F =                  F
        self.H =                  H
        self.Q =                  Q
        self.R =                  R
        self.KalmanCoolingAmplitude       = KalmanCoolingAmplitude
        self.KalmanCoolingTimeDelay       = KalmanCoolingTimeDelay
        self.KalmanAmplitudeControlWindow = KalmanAmplitudeControlWindow

        
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

    def add_squeezing_pulses(self, PulseStartTime, PulseLength, TimeBetweenPulses, PulseDepth, NumberOfPulses):
        """
        add squeezing pulses to simulation

        PulseStartTime : float
            time at which to start pulse
        PulseLength : float
            duration of pulse, associated with Omega1, angular frequency 
            of particle at lower trap power TrapPower*(1-PulseDepth)
        TimeBetweenPulses : float
            time duration between pulses, associated with Omega0, angular frequency 
            of particle at higher trap power TrapPower*(1)
        PulseDepth : float
            normalised pulse depth - i.e. if trapping power is P power is
            P*(1-PulseDepth) during pulse
        NumberOfPulses : integer
            how many squeezing pulses to perform

        """
        self.SqueezingPulseArray = generate_pulse_array(self.tArray.get_array(), PulseStartTime, PulseLength, TimeBetweenPulses, PulseDepth, NumberOfPulses)
        return None

    def add_optimal_squeezing_pulses(self, PulseStartTime, PulseDepth, NumberOfPulses):
        """
        add optimal squeezing pulses to simulation (Time for pulse is 1/4 of a trap
        cycle at lower power, time between pulses is 1/4 of a trap cycle at higher
        power.) Trap frequency during pulse is sqrt(1-PulseDepth)*TrapFrequency where
        Trap frequency is trap frequency at higher power.

        PulseStartTime : float
            time at which to start pulse
        PulseDepth : float
            normalised pulse depth - i.e. if trapping power is P power is
            P*(1-PulseDepth) during pulse
        NumberOfPulses : integer
            how many squeezing pulses to perform

        """
        freq0 = self.Omega0/(2*np.pi)
        self.TimeBetweenPulses = 0.25*(1/freq0)
        Omega1 = np.sqrt(1-PulseDepth)*self.Omega0
        freq1 = Omega1/(2*np.pi)
        self.PulseLength = 0.25*(1/freq1)
        self.SqueezingPulseArray = generate_pulse_array(self.tArray.get_array(), PulseStartTime, self.PulseLength, self.TimeBetweenPulses, PulseDepth, NumberOfPulses)
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
        self.q, self.v, self.qest, self.vest, self.KalmanFeedbackArray = solver(self.q,
                                                      self.v,
                                                      float(self.dt),
                                                      self.dwArray,
                                                      float(self.Gamma0),
                                                      float(self.deltaGamma),
                                                      float(self.Omega0),
                                                      float(self.b_v),
                                                      float(self.alpha),
                                                      float(self.beta),
                                                      float(self.DoubleFreqAmplitude),
                                                      float(self.DoubleFreqPhaseDelay),
                                                      float(self.SingleFreqAmplitude),
                                                      float(self.SingleFreqPhaseDelay),
                                                      float(self.liadTau),
                                                      SqueezingPulseArray=self.SqueezingPulseArray,
                                                      startIndex=startIndex,
                                                      NumTimeSteps=NumTimeSteps,
                                                      mass = self.mass,
                                                      filter_sample_freq = self.filter_sample_freq,
                                                      x_initial =        self.x_initial,
                                                      P_initial =        self.P_initial,
                                                      F =                self.F,
                                                      H =                self.H,
                                                      Q =                self.Q,
                                                      R =                self.R,
                                                      KalmanCoolingAmplitude       = self.KalmanCoolingAmplitude,
                                                      KalmanCoolingTimeDelay       = self.KalmanCoolingTimeDelay,
                                                      KalmanAmplitudeControlWindow = self.KalmanAmplitudeControlWindow,
        )
        
        return self.q, self.v

#def _a_v(q, v, Gamma0, Omega0, eta):
#    return -(Gamma0 + deltaGamma)*v - Omega0**2*q


def generate_pulse_array(tArray, PulseStartTime, T_Pulse, T_Between, PulseDepth, NoOfPulses):
    """
    Function for generating pulses.

    Parameters
    ----------
    tArray : ndarray
         array of time values
    PulseStartTime : float
         time at which to start pulse
    T_Pulse : float
         duration of pulse, associated with Omega1, angular frequency 
         of particle at lower trap power TrapPower*(1-PulseDepth)
    T_Between : float
         time duration between pulses, associated with Omega0, angular frequency 
         of particle at higher trap power TrapPower*(1)
    PulseDepth : float
         normalised pulse depth - i.e. if trapping power is P power is
         P*(1-PulseDepth) during pulse
    NoOfPulses : integer
         how many squeezing pulses to perform
         
    """
    n = 0
    NPulsesHappened = 0
    ValArray = np.zeros_like(tArray)
    for n, t in enumerate(tArray):
        if NPulsesHappened <= NoOfPulses - 1:
            if t <= PulseStartTime:
                ValArray[n] = 1
            elif t > PulseStartTime and t <= PulseStartTime + T_Pulse:
                ValArray[n] = 1 - PulseDepth
            elif t > PulseStartTime + T_Pulse and t < T_Pulse + T_Between:
                ValArray[n] = 1
            else:
                NPulsesHappened += 1
                PulseStartTime = PulseStartTime + T_Pulse + T_Between                
                ValArray[n] = 1
        else:
            ValArray[n] = 1
    return ValArray
