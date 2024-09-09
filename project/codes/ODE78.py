import numpy as np

def ode78(func, tspan, y0, h):
    """
    7th/8th Order Runge-Kutta method (ODE78) to solve ODEs.
    
    Parameters:
    func  - function defining the ODE (dy/dx = func(x, y))
    tspan - tuple (t0, tf) specifying the time range
    y0    - initial condition
    h     - step size
    
    Returns:
    results - matrix of time points and solution values
    """
    t0, tf = tspan
    t = t0
    y = np.array(y0)
    
    # Butcher tableau coefficients for ODE78
    alpha = np.array([0, 1/18, 1/6, 2/9, 2/3, 1, 8/9, 1])
    beta = [
        [],  # No coefficients for the first row
        [1/18],
        [-1/12, 1/4],
        [-2/81, 4/27, 8/81],
        [40/33, -4/11, -56/11, 54/11],
        [-369/73, 72/73, 5380/219, -12285/584, 2695/1752],
        [-8716/891, 656/297, 39520/891, -416/11, 52/27, 0],
        [3015/256, -9/4, -4219/78, 5985/128, -539/384, 0, 693/3328]
    ]
    
    chi = np.array([3/80, 0, 4/25, 243/1120, 77/160, 73/700, 0, 0])
    psi = np.array([57/640, 0, -16/65, 1377/2240, 121/320, 0, 891/8320, 2/35])

    # Arrays to store time and solution values
    t_values = [t]
    y_values = [y]
    
    while t < tf:
        if t + h > tf:
            h = tf - t
        
        # Initialize k array
        k = np.zeros((8, len(y0)))
        
        # Compute k values
        for i in range(8):
            sum_beta_k = np.zeros_like(y)
            for j in range(i):
                sum_beta_k += beta[i][j] * k[j]
            k[i] = h * func(t + alpha[i] * h, y + sum_beta_k)
        
        # Update y using the chi coefficients
        y = y + np.sum([chi[i] * k[i] for i in range(8)], axis=0)
        
        # Estimate error (optional, if psi coefficients are given)
        error_est = np.sum([psi[i] * k[i] for i in range(8)], axis=0)
        
        # Update time
        t += h
        
        # Store the results
        t_values.append(t)
        y_values.append(y)
    
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
    h = 10  # Step size 

    def func(t, y):
        return satellite_motion(t, y, mu)

    results = ode78(func, tspan, y0, h)
    
    print("Results (Time | Position X, Y, Z | Velocity X, Y, Z):")
    print(f"{'Time':^8} {'PosX':^12} {'PosY':^12} {'PosZ':^12} {'VelX':^12} {'VelY':^12} {'VelZ':^12}")
    print("-" * 80)
    for row in results:
        print(f"{row[0]:^8.2f} {row[1]:^12.2f} {row[2]:^12.2f} {row[3]:^12.2f} {row[4]:^12.2f} {row[5]:^12.2f} {row[6]:^12.2f}")

if __name__ == "__main__":
    main()
