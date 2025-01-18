

import numpy as np
from scipy.integrate import solve_ivp

def ODE113(ode, tspan, y0, rtol=1e-3, atol=1e-6, hmax=None, hmin=None, *args):
    """
    Solve non-stiff differential equations using a variable-order Adams-Bashforth-Moulton method.

    Parameters:
        ode: callable
            Function to evaluate the derivative (dy/dt = f(t, y)).
        tspan: array-like
            Time span for integration. Can be [t0, tfinal] or an array of specific time points.
        y0: array-like
            Initial state of the system.
        rtol: float
            Relative tolerance for the solver.
        atol: float
            Absolute tolerance for the solver.
        hmax: float, optional
            Maximum step size.
        hmin: float, optional
            Minimum step size.
        args: tuple
            Additional arguments to pass to the ode function.

    Returns:
        t: ndarray
            Time points where the solution was evaluated.
        y: ndarray
            Solution values at each time point.
    """
    # Define time span and evaluation points
    if len(tspan) == 2:
        t_eval = None  # Solver will determine points
    else:
        t_eval = tspan  # Specific time points

    # Prepare step size options
    options = {}
    if hmax:
        options['max_step'] = hmax
    if hmin:
        options['min_step'] = hmin

    # Solve the system
    result = solve_ivp(
        fun=ode,
        t_span=(tspan[0], tspan[-1]),
        y0=y0,
        method='LSODA',  # LSODA switches between Adams and BDF methods
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        **options,
        args=args
    )

    # Return results as a NumPy array
    return np.column_stack((result.t, result.y.T))  # Combine time and solution into one array




def satellite_motion(t, y, mu):
    """Satellite motion model differential equation."""
    r = np.array(y[:3])
    v = np.array(y[3:])
    r_norm = np.linalg.norm(r)
    a = -mu / r_norm**3 * r
    dydt = np.concatenate((v, a))
    return dydt
