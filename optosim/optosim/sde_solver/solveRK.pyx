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
            double alpha,
            double beta,
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
        Enviromental damping parameter (angular frequency - radians/s) - appears as (-Gamma*v) term in the SDE
    deltaGamma : float
        damping due to other effects (e.g. feedback cooling) (radians/s) - appears as (-deltaGamma*q**2*v)*dt term in the SDE
    Omega0 : float
        Trapping frequency (angular frequency - radians/s)
    eta : float
        modulation depth (as a fraction)
    b_v : float
        term multiplying the Weiner process in the SDE sqrt(2*Γ0*kB*T0/m)
    alpha : float
        prefactor multiplying the q**3 non-linearity term shows up as ([alpha*q]**3*dt) in the SDE
    beta : float
        prefactor multiplying the q**5 non-linearity term shows up as ([beta*q]**5*dt) in the SDE
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
    #print(dt, Gamma0, deltaGamma, Omega0, b_v, alpha, beta)
    cdef int n
    cdef float vK1
    cdef float qK1
    cdef float vh
    cdef float qh
    cdef float vK2
    cdef float qK2
    
    for n in range(startIndex, NumTimeSteps+startIndex):
        # stage 1 of 2-stage Runge Kutta
        vK1 = (-(Gamma0 + deltaGamma*q[n]**2)*v[n] - SqueezingPulseArray[n]*Omega0**2*q[n] + (alpha*q[n])**3 - (beta*q[n])**5)*dt + b_v*(dwArray[n] + (dt**0.5))
        qK1 = v[n]*dt
        
        vh = v[n] + vK1
        qh = q[n] + qK1

        # stage 2 of 2-stage Runge Kutta
        vK2 = (-(Gamma0 + deltaGamma*qh**2)*vh - SqueezingPulseArray[n]*Omega0**2*qh + (alpha*qh)**3 - (beta*qh)**5)*dt + b_v*(dwArray[n] - (dt**0.5))
        qK2 = vh*dt

        # update
        v[n+1] = v[n] + 0.5*(vK1 + vK2)
        q[n+1] = q[n] + 0.5*(qK1 + qK2)
        
    return q, v


