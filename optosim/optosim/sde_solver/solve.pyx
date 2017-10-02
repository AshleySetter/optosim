cimport numpy as np
cimport cython

@cython.boundscheck(False) # Turns off IndexError type warnings - e.g. a = [1, 2, 3]; a[5]
@cython.wraparound(False) # Turns off Python a[-1] indexing - will segfault.
@cython.overflowcheck(False) # Check that no integer overflows occur with arithmetic operations
@cython.initializedcheck(False) # Checks that memory is initialised when passed in
cpdef solve(np.ndarray[double, ndim=1] q,
            np.ndarray[double, ndim=1] v,
            double dt,
            np.ndarray[double, ndim=1] dwArray,
            double Gamma0,
            double deltaGamma,
            double Omega0,
            double b_v,
            np.ndarray[double, ndim=1] SqueezingPulseArray,
            int startIndex,
            int NumTimeSteps ):
    """
    Solves the SDE specified in sde_solver.py using cythonized python code.
    
    Parameters
    ----------
    q : ndarray
        intialised array of positions (with q[0] = intial position)
    v : ndarray
        intialised array of velocities (with v[0] = intial velocity)
    dt : float
        time interval / step for simulation / solver on which to solve SDE
    dwArray : ndarray
        random values to use for Weiner process
    Gamma0 : float
        Enviromental damping parameter (angular frequency - radians/s)
    deltaGamma : float
        damping due to other effects (e.g. feedback cooling) (radians/s)
    Omega0 : float
        Trapping frequency (angular frequency - radians/s)
    eta : float
        modulation depth (as a fraction)
    b_v : float
        term multiplying the Weiner process in the SDE sqrt(2*Γ0*kB*T0/m)
    SqueezingPulseArray : ndarray
        Array of values containing modulation depth of squeezing pulses (as a decimal i.e. 1 = 100%)
    startIndex : int
        array index (of q and v) at which to start solving the SDE
    NumTimeSteps : int
        The length of the time array minus 1 (number of time points over which
        to solve the SDE)

    Returns
    -------
    q : ndarray
        array of positions with time found from solving the SDE
    v : ndarray
        array of velocities with time found from solving the SDE
    """
    cdef int n
    for n in range(startIndex, NumTimeSteps+startIndex):
        v[n+1] = v[n] + (-(Gamma0 + deltaGamma)*v[n] - SqueezingPulseArray[n]*Omega0**2*q[n])*dt + b_v*dwArray[n]
        q[n+1] = q[n] + v[n]*dt
    return q, v


