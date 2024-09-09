import numpy as np

def rk4(func, tspan, y0, h):
    """
    Fourth-order Runge-Kutta method to solve ODEs.
    
    Parameters:
    func  - function defining the ODE (dy/dx = func(x, y))
    tspan - tuple (t0, tf) specifying the time range
    y0    - initial condition
    h     - step size
    
    Returns:
    results - matrix of time points and solution values
    """
    t0, tf = tspan
    t_values = np.arange(t0, tf + h, h)
    y_values = np.zeros((len(t_values), len(y0)))
    y = np.array(y0)
    
    for i, t in enumerate(t_values):
        y_values[i] = y
        k1 = h * func(t, y)
        k2 = h * func(t + h/2, y + k1/2)
        k3 = h * func(t + h/2, y + k2/2)
        k4 = h * func(t + h, y + k3)
        y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    results = np.column_stack((t_values, y_values))  # Combine time and state values
    return results

def satellite_motion(t, y, mu):
    """
    Satellite motion model differential equation.
    
    Parameters:
    t  - time (not used in this model, but required for ODE solver)
    y  - vector containing position (r) and velocity (v) [r_x, r_y, r_z, v_x, v_y, v_z]
    mu - standard gravitational parameter
    
    Returns:ٍ
    dydt - derivative of the state vector [v_x, v_y, v_z, a_x, a_y, a_z]
    """
    r = np.array(y[:3])  # Position vector r = [r_x, r_y, r_z]
    v = np.array(y[3:])  # Velocity vector v = [v_x, v_y, v_z]
    
    r_norm = np.linalg.norm(r)  # Magnitude of the position vector
    
    a = -mu / r_norm**3 * r  # Acceleration vector
    
    dydt = np.concatenate((v, a))  # Combine velocity and acceleration
    
    return dydt

def main():
    tspan = (0, 3600)  # Time range from 0 to 10 units
    r0 = np.array([-3829.29, 5677.86, -1385.16])  # Initial position vector (example)
    v0 = np.array([-1.69535, -0.63752, 7.33375])   # Initial velocity vector (example)
    y0 = np.concatenate((r0, v0))  # Initial state vector
    mu = 398600  # Standard gravitational parameter (example for Earth)
    h = 0.1     # Step size 

    def func(t, y):
        return satellite_motion(t, y, mu)

    results = rk4(func, tspan, y0, h)
    
    print("Results (Time | Position X, Y, Z | Velocity X, Y, Z):")
    print(f"{'Time':^8} {'PosX':^12} {'PosY':^12} {'PosZ':^12} {'VelX':^12} {'VelY':^12} {'VelZ':^12}")
    print("-" * 80)
    for row in results:
        print(f"{row[0]:^8.2f} {row[1]:^12.2f} {row[2]:^12.2f} {row[3]:^12.2f} {row[4]:^12.2f} {row[5]:^12.2f} {row[6]:^12.2f}")

if __name__ == "__main__":
    main()
