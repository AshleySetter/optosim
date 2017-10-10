from optoanalysis import DataObject
from optosim.sde_solver import sde_solver
import numpy as _np
from multiprocessing import Pool as _Pool
from frange import frange

class SimData(DataObject):
    """
    Simualtes a particle by solving the SDE for a single degree of freedom for each 
    trap frequency value in TrapFreqArray (and values of eta [modulation depth] in 
    etaArray and using the values specified by the other parameters to this object 
    and then convolving the signals from each by addition in time-space along with 
    addition of white noise produced by numpy.random.normal with a std deviation
    given by NoiseStdDev. 

    Attributes
    ----------
    q0 : float
        
    v0 : float
        
    TimeTuple : tuple
        
    timeStart : float
        
    timeEnd : float
        
    SampleFreq : float
        
    TrapFreqArray : ndarray
        
    Gamma0 : float
        Enviromental damping
    mass : float
        mass of nanoparticle (in Kg)
    ConvFactor : float
        
    NoiseStdDev : float
        
    T0 : float, optional
        
    deltaGammaArray : ndarray
        array containing the additional damping on each degree of freedom 
        due to other effects (e.g. feedback cooling)

    dt : float
        
    seed : float
        random seed for generating the weiner paths for SDE solving
        defaults to None i.e. no seeding of random numbers
        sets the seed prior to initialising the SDE solvers such
        that the data is repeatable but that each solver uses
        different random numbers
    dtSample : float
        




    DownSampleAmount : int
        
    timeStep : float
        
    time : ndarray
        
    simtime : frange
        
    Noise : ndarray
        
    voltage : ndarray
        
    TrueSignals : ndarray
        



    Omega0 : float
        Trapping frequency
    Gamma0 : float
        Enviromental damping
    mass : float

    eta : float, optional
        modulation depth (as a fraction), defaults to 0
    T0 : float, optional
        Temperature of the environment, defaults to 300

    q0 : float, optional
        initial position, defaults to 0
    v0 : float, optional
        intial velocity, defaults to 0

    dt : float, optional
        time interval for simulation / solver
    seed : float, optional
        random seed for generate_weiner_path, defaults to None
        i.e. no seeding of random numbers

    """
    def __init__(self, TimeTuple, SampleFreq, TrapFreqArray, Gamma0, mass, ConvFactorArray, NoiseStdDev, T0=300.0, deltaGammaArray=None, dt=1e-9, seed=None, NPerSegmentPSD=1000000):
        """
        Parameters
        ----------
        TimeTuple : tuple, optional
            tuple of start and stop time for simulation / solver
        SampleFreq : float
            Sample freq to downsample data to (Hz)
        TrapFreqArray : ndarray
            Array of trap frequencies of Z, X and Y motion (Hz).
        Gamma0 : float
            Damping due to the enviroment (radians/second)
        mass : float
            mass of the nanoparticle (kg)
        ConvFactorArray : ndarray
            Conversion factors to use to go from motion in nms
            to signal in volts for each degree of freedom
        NoiseStdDev : float
            std deviation of white noise applied to particle signal
            after generation from SDE solving.
        T0 : float, optional
            Temperature of the environment
        deltaGammaArray : ndarray, optional
            array containing the additional damping on each degree of freedom 
            due to other effects (e.g. feedback cooling)
        dt : float, optional
            time step for SDE solver
        seed : float, optional
            random seed for generating the weiner paths for SDE solving
            defaults to None i.e. no seeding of random numbers
            sets the seed prior to initialising the SDE solvers such
            that the data is repeatable but that each solver uses
            different random numbers. No seed by default.
        NPerSegnmentPSD : int, optional
            number of points per segment to use in calculating the PSD


        """
        self.q0 = 0.0
        self.v0 = 0.0
        self.TimeTuple = (TimeTuple[0], TimeTuple[1])
        self.timeStart = TimeTuple[0]
        self.timeEnd = TimeTuple[1]
        self.SampleFreq = SampleFreq
        self.TrapFreqArray = _np.array(TrapFreqArray)
        self.Gamma0 = Gamma0
        if deltaGammaArray == None:
            self.deltaGammaArray = _np.zeros_like(TrapFreqArray)
        else:
            if len(deltaGammaArray) != len(TrapFreqArray):
                raise("deltaGammaArray should be the same length as TrapFreqArray")
            self.deltaGammaArray = deltaGammaArray
        self.mass = mass
        self.ConvFactorArray = ConvFactorArray
        self.NoiseStdDev = NoiseStdDev
        self.T0 = T0
        self.dt = dt
        self.seed = seed        
        dtSample = 1/SampleFreq
        self.DownSampleAmount = round(dtSample/dt)
        self.timeStep = dtSample/dt
        if _np.isclose(dtSample/dt, self.DownSampleAmount, atol=1e-6) == False:
            raise ValueError("The sample rate {} has a time interval between samples of {}, this is not a multiple of the simualted time interval {}. dtSample/dt = {}".format(SampleFreq, dtSample, dt, self.timeStep))
        self.generate_simulated_data() # solves SDE for each frequency and eta value specified
        # along requested time interval
        self.time = frange(TimeTuple[0], TimeTuple[1], self.DownSampleAmount*dt)
        self.simtime = frange(TimeTuple[0], TimeTuple[1], dt)
        self.Noise = _np.random.normal(0, self.NoiseStdDev, len(self.time))
        self.voltage = _np.copy(self.Noise)
        self.TrueSignals = []
        for i, sdesolver in enumerate(self.sde_solvers):
            self.TrueSignals.append(_np.array([sdesolver.q, sdesolver.v]))
            self.voltage += ConvFactorArray[i]*sdesolver.q[::self.DownSampleAmount]
        self.TrueSignals = _np.array(self.TrueSignals)
        self.get_PSD(NPerSegmentPSD)
        del(self.sde_solvers)
        return None

    def generate_simulated_data(self):
        self.sde_solvers = []
        if self.seed != None:
            _np.random.seed(self.seed)
        for i, freq in enumerate(self.TrapFreqArray):
            TrapOmega = freq*2*_np.pi
            solver = sde_solver.sde_solver(TrapOmega, self.Gamma0, self.deltaGammaArray[i], self.mass, T0=self.T0, q0=self.q0, v0=self.v0, TimeTuple=self.TimeTuple, dt=self.dt)
            self.sde_solvers.append(solver)
        #workerPool = _Pool()
        #workerPool.map(run_solve, self.sde_solvers)
        for solver in self.sde_solvers:
            print('solving...')
            solver.solve()
        return None

def run_solve(sde_solver):
    print('solving...')
    sde_solver.q, sde_solver.v = sde_solver.solve()
    return None

