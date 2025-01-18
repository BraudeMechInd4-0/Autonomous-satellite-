import numpy as np

import numpy as np

def rk45_step(func, t, y, h, rtol, atol):
    """
    Perform a single RK45 step using the new coefficients from the Butcher tableau.
    
    Parameters:
    func: function
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
    y_next: array-like
        Next state vector.
    h_new: float
        Adjusted step size for next step.
    """
    # New coefficients from the provided Butcher tableau
    a = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    b = [
        [0],
        [1/5],
        [3/40, 9/40],
        [44/45, -56/15, 32/9],
        [19372/6561, -25360/2187, 64448/6561, -212/729],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
    ]
    c4 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]  # 4th order solution
    c5 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]  # 5th order solution
    
    # Perform the RK45 step using the new coefficients
    k1 = h * np.array(func(t, y))
    k2 = h * np.array(func(t + a[1]*h, y + b[1][0]*k1))
    k3 = h * np.array(func(t + a[2]*h, y + b[2][0]*k1 + b[2][1]*k2))
    k4 = h * np.array(func(t + a[3]*h, y + b[3][0]*k1 + b[3][1]*k2 + b[3][2]*k3))
    k5 = h * np.array(func(t + a[4]*h, y + b[4][0]*k1 + b[4][1]*k2 + b[4][2]*k3 + b[4][3]*k4))
    k6 = h * np.array(func(t + a[5]*h, y + b[5][0]*k1 + b[5][1]*k2 + b[5][2]*k3 + b[5][3]*k4 + b[5][4]*k5))
    k7 = h * np.array(func(t + a[6]*h, y + b[6][0]*k1 + b[6][1]*k2 + b[6][2]*k3 + b[6][3]*k4 + b[6][4]*k5 + b[6][5]*k6))
    
    # Calculate the 4th and 5th order solutions
    y4 = y + np.dot(c4, [k1, k2, k3, k4, k5, k6, k7])
    y5 = y + np.dot(c5, [k1, k2, k3, k4, k5, k6, k7])
    
    # Estimate the error
    error = np.linalg.norm(y5 - y4, ord=np.inf) / (atol + rtol * np.maximum(np.linalg.norm(y4), np.linalg.norm(y5)))
    
    # Determine the new step size
    h_new = h * min(2, max(0.1, 0.9 / error**0.2)) if error != 0 else h * 2
    
    return t + h, y5, h_new

# Remaining code (main, satellite_motion, etc.) remains unchanged.

def ode45(func, t_span, y0, rtol=1e-3, atol=1e-6):
    """
    Solve an ODE using the RK45 method with the given time segments.

    Parameters:
    func: function
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
            t_next, y_next, h = rk45_step(func, t, y, h, rtol, atol)
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
    tspan = [0, 300]  # Time range from 0 to 10000 seconds
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

    # Run ODE45 with Gauss-Lobatto points
    result = ode45(func, gauss_lobatto_tspan, y0, rtol=1e-8, atol=1e-10)

    # Print results in organized format
    print("Results (Time | Position X, Y, Z | Velocity X, Y, Z):")
    print(f"{'Time':^8} {'PosX':^12} {'PosY':^12} {'PosZ':^12} {'VelX':^12} {'VelY':^12} {'VelZ':^12}")
    print("-" * 80)
    
    for row in result:
        print(f"{row[0]:^8.2f} {row[1]:^12.2f} {row[2]:^12.2f} {row[3]:^12.2f} {row[4]:^12.2f} {row[5]:^12.2f} {row[6]:^12.2f}")

if __name__ == "__main__":
    main()
