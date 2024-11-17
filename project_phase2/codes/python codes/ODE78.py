import numpy as np
import coefficients78 as coeff


def rk78_step(ode_func, t, y, h, rtol, atol):
    """
    Perform a single RK78 step using the 13-stage Runge-Kutta 8(7) coefficients.

    Parameters:
    ode_func: function
        Function to integrate (dy/dt = func(t, y)).
    t: float
        Current time.
    y: array-like
        Current state vector.
    h: float
        Step size.
    rtol: float
        Relative tolerance for error control.
    atol: float
        Absolute tolerance for error control.

    Returns:
    t_next: float
        Next time step.
    y8: array-like
        8th-order state vector (higher order solution).
    h_new: float
        Adjusted step size for next step.
    """
    # Load coefficients (a, c, b, and bh from external source/module if needed)
    c = coeff.c  # RK78 nodes
    a = coeff.a  # RK78 coupling coefficients (Butcher tableau)
    b = coeff.b  # 8th-order weights
    bh = coeff.bh  # 7th-order weights

    # Initialize stages (k1 to k13)
    k = np.zeros((len(y), 13))

    # Perform the RK78 step using the coefficients
    k[:, 0] = h * np.array(ode_func(t, y))
    
    for i in range(1, 13):
        y_temp = y.copy()
        for j in range(i):
            y_temp += a[i + 1].get(j + 1, 0) * k[:, j]
        k[:, i] = h * np.array(ode_func(t + c[i + 1] * h, y_temp))

    # Compute the 8th-order solution (y8) using the b coefficients
    y8 = y + np.dot(list(b.values()), k.T)

    # Compute the 7th-order solution (y7) using the bh coefficients
    y7 = y + np.dot(list(bh.values()), k.T)

    # Estimate the error between the 7th and 8th order solutions
    error = np.linalg.norm(y8 - y7, ord=np.inf) / (atol + rtol * np.maximum(np.linalg.norm(y7), np.linalg.norm(y8)))

    # Adjust step size based on the error estimate
    if error != 0:
        h_new = h * min(2, max(0.1, 0.9 / error**0.2))
    else:
        h_new = h * 2  # Double the step size if error is zero

    return t + h, y8, h_new




def ode78(ode_func, t_span, y0, rtol=1e-3, atol=1e-6):
    """
    Solve an ODE using the RK78 method with adaptive time stepping.

    Parameters:
    ode_func: function
        Function to integrate (dy/dt = func(t, y)).
    t_span: array-like
        Array of times at which to solve the system.
    y0: array-like
        Initial conditions.
    rtol: float
        Relative tolerance for error control.
    atol: float
        Absolute tolerance for error control.

    Returns:
    np.column_stack((tout, yout)): array-like
        A 2D array with time and corresponding state values.
    """
    
    # If only one time point is provided, create a small range around it
    if len(t_span) == 1:
        t_start = t_span[0] - 0.99 * t_span[0]
        t_end = t_span[0] + 0.05
        t_span = np.arange(t_start, t_end, 0.1)
    
    tout = [t_span[0]]  # List to store time points
    yout = [y0]  # List to store state vectors (position and velocity)
    
    t = t_span[0]
    y = np.array(y0)

    for i in range(1, len(t_span)):
        h = t_span[i] - t_span[i - 1]  # Step size as the difference between consecutive Gauss-Lobatto points
        while t < t_span[i]:
            t_next, y_next, h = rk78_step(ode_func, t, y, h, rtol, atol)
            t = t_next
            y = y_next
        
        # Append the time and state vector for the Gauss-Lobatto time points
        tout.append(t)
        yout.append(y)
    
    # Convert the time and state lists into numpy arrays
    tout = np.array(tout)
    yout = np.array(yout)

    # Return the results in the format np.column_stack((tout, yout))
    return np.column_stack((tout, yout))

def satellite_motion(t, y, mu):
    r = np.array(y[:3])  # Position vector
    v = np.array(y[3:])  # Velocity vector
    
    r_norm = np.linalg.norm(r)  # Magnitude of the position vector
    a = -mu / r_norm**3 * r  # Acceleration vector
    
    dydt = np.concatenate((v, a))  # Combine velocity and acceleration
    return dydt

def gauss_lobatto_points(n, a, b):
    """Generate Gauss-Lobatto points for the interval [a, b]."""
    points = [-np.cos(np.pi * i / (n - 1)) for i in range(n)]
    scaled_points = 0.5 * (b - a) * (np.array(points) + 1) + a
    return scaled_points

def main():
    tspan = [0, 12499.81]  # Time range from 0 to 300 seconds
    r0 = np.array([-3.31458372e+03, 2.80902481e+03, -5.40046222e+03])  # Initial position vector
    v0 = np.array([-3.42701791e+00, -6.62341508e+00, -1.34238849e+00])  # Initial velocity vector
    y0 = np.concatenate((r0, v0))  # Initial state vector
    mu = 398600  # Standard gravitational parameter (Earth)
    
    total_time = 300
    num_segments = 16
    n_points = 32
    t_start = 0
    t_end = tspan[-1]
    
    # Calculate Gauss-Lobatto points for the given time range
    gauss_lobatto_tspan = gauss_lobatto_points(n_points, t_start, t_end)
    
    def func(t, y):
        return satellite_motion(t, y, mu)

    # Run RK78 with Gauss-Lobatto points
    result = ode78(func, gauss_lobatto_tspan, y0, rtol=1e-8, atol=1e-10)

    # Print results in organized format
    print("Results (Time | Position X, Y, Z | Velocity X, Y, Z):")
    print(f"{'Time':^8} {'PosX':^12} {'PosY':^12} {'PosZ':^12} {'VelX':^12} {'VelY':^12} {'VelZ':^12}")
    print("-" * 80)
    
    for row in result:
        print(f"{row[0]:^8.2f} {row[1]:^12.2f} {row[2]:^12.2f} {row[3]:^12.2f} {row[4]:^12.2f} {row[5]:^12.2f} {row[6]:^12.2f}")

if __name__ == "__main__":
    main()
