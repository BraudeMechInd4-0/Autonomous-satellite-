import numpy as np
from scipy.interpolate import interp1d

def ODE113(ode, tspan, y0, options=None, *args):
    """
    Python implementation of MATLAB's ode113 function with improved accuracy.

    Parameters:
        ode: Callable
            Function defining the differential equations y' = f(t, y).
        tspan: array_like
            Time span [t0, tfinal] for solver-determined points, or [t0, t1, ..., tfinal] for specific points.
        y0: array_like
            Initial condition for the solution.
        options: dict, optional
            Solver options, e.g., {'RelTol': 1e-9, 'AbsTol': 1e-9}.
        args: tuple
            Additional arguments to pass to the ODE function.

    Returns:
        result: ndarray
            Combined array where the first column is time and the remaining columns are solution values.
    """
    # Default solver options
    rel_tol = options.get('RelTol', 1e-9) if options else 1e-9
    abs_tol = options.get('AbsTol', 1e-9) if options else 1e-9
    hmax = options.get('hmax', 0.5) if options else 0.5
    hmin = options.get('hmin', 1e-10) if options else 1e-10

    # Determine whether tspan specifies specific points or a range
    if len(tspan) == 2:
        t_eval = None  # Solver will determine points dynamically
        t_start, t_end = tspan
    else:
        t_eval = np.array(tspan)  # Specific points provided
        t_start, t_end = tspan[0], tspan[-1]

    # Initial setup
    t = t_start
    y = np.array(y0, dtype=float)
    h = 0.01

    tout = [t]
    yout = [y]

    # Main integration loop
    while t < t_end:
        # Ensure step size does not overshoot
        if t + h > t_end:
            h = t_end - t

        # Predictor step (Adams-Bashforth)
        yp = ode(t, y, *args)
        y_pred = y + h * yp

        # Corrector step (Adams-Moulton) with multiple iterations for better accuracy
        for _ in range(3):  # Iterate the corrector 3 times
            yp_corr = ode(t + h, y_pred, *args)
            y_corr = y + h * (yp + yp_corr) / 2
            y_pred = y_corr

        # Error estimation
        err = np.linalg.norm((y_corr - y_pred) / np.maximum(abs(y), abs_tol), ord=np.inf)

        if err <= rel_tol:
            # Accept step
            t += h
            y = y_corr
            tout.append(t)
            yout.append(y)

        # Adjust step size
        if err == 0:
            h = min(h * 2, hmax)
        else:
            h = max(min(h * min(2, 0.9 * (rel_tol / err) ** 0.5), hmax), hmin)

    # Convert outputs to numpy arrays
    tout = np.array(tout)
    yout = np.array(yout)

    # Interpolation for specific points if t_eval is provided
    if t_eval is not None:
        interp_func = interp1d(tout, yout, axis=0, kind='cubic')
        y_eval = interp_func(t_eval)
        return np.column_stack((t_eval, y_eval))

    # Default return (all computed points)
    return np.column_stack((tout, yout))

def satellite_motion(t, y, mu):
    """
    Defines the satellite motion differential equation.
    """
    r = np.array(y[:3])
    v = np.array(y[3:])
    r_norm = np.linalg.norm(r)
    a = -mu / r_norm**3 * r
    return np.concatenate((v, a))

def main():
    """
    Example usage of the ODE113 implementation with satellite motion.
    """
    tspan = [0, 50, 100, 150, 200, 250, 300]  # Specific evaluation points
    r0 = [-3.31458372e+03, 2.80902481e+03, -5.40046222e+03]
    v0 = [-3.42701791e+00, -6.62341508e+00, -1.34238849e+00]
    y0 = np.concatenate((r0, v0))
    mu = 398600  # Standard gravitational parameter (Earth)

    # Run ODE113
    options = {'RelTol': 1e-9, 'AbsTol': 1e-9, 'hmax': 10, 'hmin': 0.001}
    result = ODE113(satellite_motion, tspan, y0, options, mu)

    # Print results
    print("Time | Position X, Y, Z | Velocity X, Y, Z")
    print("-" * 50)
    for row in result:
        print(f"{row[0]:.2f} | {row[1]:.3f}, {row[2]:.3f}, {row[3]:.3f} | {row[4]:.3f}, {row[5]:.3f}, {row[6]:.3f}")

if __name__ == "__main__":
    main()
