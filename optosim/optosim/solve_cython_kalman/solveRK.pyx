import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport atan2, sin

#cpdef get_z_n(n, z):
#    cdef float z_n
##    e = np.e
#    if n < 0:
#        z_n = 0
#    else:
#        z_n = z[n]
#    return z_n

#cpdef get_I_n_plus_1(In, n, M, Omega0, dt, z, np):
#    In_p1 = In + get_z_n(n, z)*np.exp(-1j*Omega0*n*dt)*dt - get_z_n(n-M, z)*np.exp(-1j*Omega0*(n-M)*dt)*dt
#    return In_p1
#
#cpdef calc_zw0_n(n, In, Omega0, dt, np):
#    zw0_n = np.exp(1j*Omega0*n*dt)*(1/dt)*In
#    return zw0_n
  
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
            double DoubleFreqAmplitude,
            double DoubleFreqPhaseDelay,
            double SingleFreqAmplitude,
            double SingleFreqPhaseDelay,
            double dTau,
            np.ndarray[double, ndim=1] SqueezingPulseArray,
            int startIndex,
            int NumTimeSteps,
            double mass,
            double filter_sample_freq,
            x_initial,
            P_initial,
            F,
            H,
            Q,
            R,
            KalmanCoolingAmplitude,
            KalmanCoolingTimeDelay,
            KalmanAmplitudeControlWindow,):
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
    cdef int M
    cdef float phi_n
    cdef float DoubleFreqFeedback
    cdef float SingleFreqFeedback

    M = int(dTau/dt)

#    I0 = q[0]
#    zw0_0 = calc_zw0_n(0, I0, Omega0, dt, np)

#    In = I0

#    phi_array = []
    
    narray = range(startIndex, NumTimeSteps+startIndex)
    S = np.random.choice([-1, 1], NumTimeSteps) # randomly choose -1 or 1 with 50% chance

    q_estimate_array = np.zeros_like(q)
    v_estimate_array = np.zeros_like(q)
    KalmanFeedbackArray = np.zeros_like(q)
    
    filter_dt = 1/filter_sample_freq

    NFilt = int(filter_dt/dt)
    if int(filter_dt/dt) != filter_dt/dt:
        print("Filter won't work correctly filter sample freq, {}, not multiple of simulation frequency, {}.".format(1/filter_dt, 1/dt))

    current_state_estimate = x_initial
    current_prob_estimate = P_initial

    q_estimate = current_state_estimate[0]
    v_estimate = current_state_estimate[1]

    KalmanCoolingIndexDelay = int(round(KalmanCoolingTimeDelay/dt))
    
    for n in narray:
        # Method 1 
        #zw0_n = calc_zw0_n(n, In, Omega0, dt, np)

        # Method 2
        #zw0_n = 0
        #for n_tmp in range(n-M, n+1):
        #    zw0_n += q[n_tmp]*np.exp(-1j*Omega0*(n_tmp)*dt)
        #zw0_n *= np.exp(1j*Omega0*n*dt)/dt

        # Method 3
        #integralTerm = q[1:]*np.exp(-1j*Omega0*np.array(list(narray))*dt)
        #zw0_n = np.trapz(integralTerm[n-M:n])
        
        #phi_n = np.angle(zw0_n)
        phi_n = atan2(v[n]/Omega0, q[n])
#        phi_array.append(phi_n)
        DoubleFreqFeedback = DoubleFreqAmplitude*sin(2*phi_n + DoubleFreqPhaseDelay)
        SingleFreqFeedback = SingleFreqAmplitude*sin(phi_n + SingleFreqPhaseDelay)
        
        # Kalman Feedback Cooling
        try:
            KalmanCoolingMultiplier = KalmanCoolingAmplitude/np.nanmax(q_estimate_array[n-1-KalmanAmplitudeControlWindow:n-1]**2)
            #print("max: {}".format(np.nanmax(q_estimate_array[n-1-KalmanAmplitudeControlWindow:n-1]**2)))
            #print("array: {}".format(q_estimate_array[n-1-KalmanAmplitudeControlWindow:n-1]**2))
        except Exception,e:
            KalmanCoolingMultiplier = 0
        #print(KalmanCoolingMultiplier)
        KalmanFeedback = KalmanCoolingMultiplier*q_estimate_array[n-1-KalmanCoolingIndexDelay]**2
        #print(KalmanFeedback)
        if np.isnan(KalmanFeedback):
            KalmanFeedback = 0
#        print(KalmanFeedback)
        KalmanFeedbackArray[n] = KalmanFeedback
        
        # stage 1 of 2-stage Runge Kutta
        vK1 = (-(Gamma0 + deltaGamma*q[n]**2)*v[n] - Omega0**2*(SqueezingPulseArray[n] + DoubleFreqFeedback + SingleFreqFeedback + KalmanFeedback)*q[n] + (alpha*q[n])**3 - (beta*q[n])**5)*dt + b_v*(dwArray[n] + S[n]*(dt**0.5))
        qK1 = (v[n])*dt 
        
        vh = v[n] + vK1
        qh = q[n] + qK1

        # stage 2 of 2-stage Runge Kutta
        vK2 = (-(Gamma0 + deltaGamma*qh**2)*vh - Omega0**2*(SqueezingPulseArray[n] + DoubleFreqFeedback + SingleFreqFeedback + KalmanFeedback)*qh + (alpha*qh)**3 - (beta*qh)**5)*dt + b_v*(dwArray[n] - S[n]*(dt**0.5))
        qK2 = (vh)*dt

        # update
        v[n+1] = v[n] + 0.5*(vK1 + vK2)
        q[n+1] = q[n] + 0.5*(qK1 + qK2)

        if n % NFilt == 0:
            #---------------------------Prediction step-----------------------------
            predicted_state_estimate = F * current_state_estimate
            predicted_prob_estimate = (F * current_prob_estimate) * np.transpose(F) + Q
            #--------------------------Observation step-----------------------------
            measurement = q[n]
            innovation = measurement - H*predicted_state_estimate
            innovation_covariance = H*predicted_prob_estimate*np.transpose(H) + R
            #-----------------------------Update step-------------------------------
            kalman_gain = predicted_prob_estimate * np.transpose(H) * np.linalg.inv(innovation_covariance)
            KG = kalman_gain
            current_state_estimate = predicted_state_estimate + kalman_gain * innovation
            # We need the size of the matrix so we can make an identity matrix.
            size = current_prob_estimate.shape[0]
            # eye(n) = nxn identity matrix.
            current_prob_estimate = (np.eye(size)-kalman_gain*H)*predicted_prob_estimate
            q_estimate = current_state_estimate[0]
            v_estimate = current_state_estimate[1]
            
        #        In = get_I_n_plus_1(In, n+1, M, Omega0, dt, q, np) # In+1 for next iteration

        q_estimate_array[n] = q_estimate
        v_estimate_array[n] = v_estimate
        
    return q, v, q_estimate_array, v_estimate_array, KalmanFeedbackArray


