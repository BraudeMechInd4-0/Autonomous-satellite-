import numpy as np

def rk4_step(f, t, y, h):
    """Performs a single step of the classic 4th-order Runge-Kutta method."""
    k1 = h * f(t, y)
    k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
    k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def ODE113(f, time_points, y0,tol, hmax=1.0,hmin=1e-6,tspan=(0,30)):
    """
    Adams-Bashforth-Moulton method using fixed time points to solve ODEs.
    
    Parameters:
    f           - function defining the ODE (dy/dx = f(t, y))
    time_points - array of specific time points
    y0          - initial condition
    
    Returns:
    t_values - array of time points
    y_values - array of solution values corresponding to time points
    """
    print("in ode113 y0=",y0)
    # If only one time point is provided, return the initial condition
    if len(time_points) == 1:
        time_points = np.arange( time_points[0]-0.99*time_points[0] ,  time_points[0]+0.05 , 0.1)
        
    t_values_list = [time_points[0]]
    y_values_list = [y0]

    # Initial conditions for the first few steps using RK4
    for i in range(3):
        if i >= len(time_points) - 1:
            break
        t = time_points[i]
        h = time_points[i + 1] - t
        y_next = rk4_step(f, t, y_values_list[-1], h)
        t_values_list.append(time_points[i + 1])
        y_values_list.append(y_next)

    # Adams-Bashforth-Moulton method over fixed time points
    for i in range(3, len(time_points) - 1):
        t = time_points[i]
        h = time_points[i + 1] - t

        # Adams-Bashforth predictor
        y_pred = y_values_list[-1] + h / 24 * (
            55 * f(time_points[i], y_values_list[-1])
            - 59 * f(time_points[i - 1], y_values_list[-2])
            + 37 * f(time_points[i - 2], y_values_list[-3])
            - 9 * f(time_points[i - 3], y_values_list[-4])
        )

        # Adams-Moulton corrector
        y_correct = y_values_list[-1] + h / 24 * (
            9 * f(time_points[i + 1], y_pred)
            + 19 * f(time_points[i], y_values_list[-1])
            - 5 * f(time_points[i - 1], y_values_list[-2])
            + f(time_points[i - 2], y_values_list[-3])
        )

        t_values_list.append(time_points[i + 1])
        y_values_list.append(y_correct)

    return np.column_stack((t_values_list, y_values_list))

def satellite_motion(t, y, mu):
    """Satellite motion model differential equation."""
    r = np.array(y[:3])
    v = np.array(y[3:])
    r_norm = np.linalg.norm(r)
    a = -mu / r_norm**3 * r
    dydt = np.concatenate((v, a))
    return dydt


