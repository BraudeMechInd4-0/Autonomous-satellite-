import numpy as np

def ode45(func, tspan, y0, tol, hmax, hmin):
    """
    Dormand-Prince method (ODE45) to solve ODEs.
    
    Parameters:
    func  - function defining the ODE (dy/dx = func(x, y))
    tspan - tuple (t0, tf) specifying the time range
    y0    - initial condition
    tol   - tolerance for adaptive step size
    hmax  - maximum step size
    hmin  - minimum step size
    
    Returns:
    results - matrix of time points and solution values
    """
    t0, tf = tspan
    t_values = []
    y_values = []
    
    # Initial conditions
    t = t0
    y = np.array(y0)
    h = hmax
    
    # Coefficients for the Dormand-Prince method
    a21 = 1.0 / 5.0
    a31 = 3.0 / 40.0
    a32 = 9.0 / 40.0
    a41 = 44.0 / 45.0
    a42 = -56.0 / 15.0
    a43 = 32.0 / 9.0
    a51 = 19372.0 / 6561.0
    a52 = -25360.0 / 2187.0
    a53 = 64448.0 / 6561.0
    a54 = -212.0 / 729.0
    a61 = 9017.0 / 3168.0
    a62 = -355.0 / 33.0
    a63 = 46732.0 / 5247.0
    a64 = 49.0 / 176.0
    a65 = -5103.0 / 18656.0
    a71 = 35.0 / 384.0
    a73 = 500.0 / 1113.0
    a74 = 125.0 / 192.0
    a75 = -2187.0 / 6784.0
    a76 = 11.0 / 84.0
    
    c2 = 1.0 / 5.0
    c3 = 3.0 / 10.0
    c4 = 4.0 / 5.0
    c5 = 8.0 / 9.0
    c6 = 1.0
    c7 = 1.0
    
    b1 = 35.0 / 384.0
    b3 = 500.0 / 1113.0
    b4 = 125.0 / 192.0
    b5 = -2187.0 / 6784.0
    b6 = 11.0 / 84.0
    
    b1p = 5179.0 / 57600.0
    b3p = 7571.0 / 16695.0
    b4p = 393.0 / 640.0
    b5p = -92097.0 / 339200.0
    b6p = 187.0 / 2100.0
    b7p = 1.0 / 40.0
    
    while t < tf:
        t_values.append(t)
        y_values.append(y)
        
        # Compute function values
        K1 = func(t, y)
        K2 = func(t + c2 * h, y + h * (a21 * K1))
        K3 = func(t + c3 * h, y + h * (a31 * K1 + a32 * K2))
        K4 = func(t + c4 * h, y + h * (a41 * K1 + a42 * K2 + a43 * K3))
        K5 = func(t + c5 * h, y + h * (a51 * K1 + a52 * K2 + a53 * K3 + a54 * K4))
        K6 = func(t + h, y + h * (a61 * K1 + a62 * K2 + a63 * K3 + a64 * K4 + a65 * K5))
        K7 = func(t + h, y + h * (a71 * K1 + a73 * K3 + a74 * K4 + a75 * K5 + a76 * K6))
        
        # Compute the 4th-order and 5th-order solutions
        y4th = y + h * (b1 * K1 + b3 * K3 + b4 * K4 + b5 * K5 + b6 * K6)
        y5th = y + h * (b1p * K1 + b3p * K3 + b4p * K4 + b5p * K5 + b6p * K6 + b7p * K7)
        
        # Estimate the error
        error = np.linalg.norm(y5th - y4th)
        # Add a small epsilon to avoid division by zero
        epsilon = 1e-10
        # Calculate the scaling factor
        delta = 0.84 * (tol / epsilon) ** (1.0 / 5.0)
        
        if error < tol:
            t += h
            y = y5th  # Update y with the higher-order estimate (5th-order)
        
        # Adjust step size based on delta
        if delta <= 0.1:
            h *= 0.1
        elif delta >= 4.0:
            h *= 4.0
        else:
            h *= delta
        
        # Ensure the step size stays within bounds
        h = max(min(h, hmax), hmin)
        
        # Adjust final step if it overshoots
        if t + h > tf:
            h = tf - t
    
    # Convert results to numpy arrays
    t_values = np.array(t_values)
    y_values = np.array(y_values)
    
    results = np.column_stack((t_values, y_values))  # Combine time and state values
    return results

def satellite_motion(t, y, mu):
    """
    Satellite motion model differential equation.
    
    Parameters:
    t  - time (not used in this model, but required for ODE solver)
    y  - vector containing position (r) and velocity (v) [r_x, r_y, r_z, v_x, v_y, v_z]
    mu - standard gravitational parameter
    
    Returns:
    dydt - derivative of the state vector [v_x, v_y, v_z, a_x, a_y, a_z]
    """
    r = np.array(y[:3])  # Position vector r = [r_x, r_y, r_z]
    v = np.array(y[3:])  # Velocity vector v = [v_x, v_y, v_z]
    
    r_norm = np.linalg.norm(r)  # Magnitude of the position vector
    
    a = -mu / r_norm**3 * r  # Acceleration vector
    
    dydt = np.concatenate((v, a))  # Combine velocity and acceleration
    
    return dydt

def main():
    tspan = (0, 3600)  # Time range from 0 to 3600 seconds
    r0 = np.array([-3829.29, 5677.86, -1385.16])  # Initial position vector (example)
    v0 = np.array([-1.69535, -0.63752, 7.33375])   # Initial velocity vector (example)
    y0 = np.concatenate((r0, v0))  # Initial state vector
    mu = 398600  # Standard gravitational parameter (example for Earth)
    
    tol = 1e-6   # Tolerance for adaptive step size
    hmax = 100   # Maximum step size
    hmin = 1e-6  # Minimum step size

    def func(t, y):
        return satellite_motion(t, y, mu)

    results = ode45(func, tspan, y0, tol, hmax, hmin)
    
    print("Results (Time | Position X, Y, Z | Velocity X, Y, Z):")
    print(f"{'Time':^8} {'PosX':^12} {'PosY':^12} {'PosZ':^12} {'VelX':^12} {'VelY':^12} {'VelZ':^12}")
    print("-" * 80)
    for row in results:
        print(f"{row[0]:^8.2f} {row[1]:^12.2f} {row[2]:^12.2f} {row[3]:^12.2f} {row[4]:^12.2f} {row[5]:^12.2f} {row[6]:^12.2f}")

if __name__ == "__main__":
    main()
