import numpy as np
from numpy.polynomial.legendre import leggauss
import time

def rk4(odefun, t_gauss_lobatto, y0,h=0.1):
    """
    Fixed step 4th order Runge-Kutta method (RK4) using Gauss-Lobatto points.

    Parameters:
        odefun - function to integrate (dy/dx = f(x))
        tspan  - time span for integration (not used if Gauss-Lobatto points are provided)
        y0     - initial conditions at t = tspan[0]
        t_gauss_lobatto - Gauss-Lobatto time points for integration
        
    Returns:
        tout - time points of the solution
        yout - solution at each time point
    """
    # If only one time point is provided, return the initial condition
    if len(t_gauss_lobatto) == 1:
        t_gauss_lobatto = np.arange( t_gauss_lobatto[0]-0.99*t_gauss_lobatto[0] ,  t_gauss_lobatto[0]+0.05 , 0.1)
        
    tout = np.array(t_gauss_lobatto)
    yout = np.zeros((len(tout), len(y0)))

    # Initialize the first value of yout with the initial condition
    y = np.array(y0)
    yout[0, :] = y
    
    
    
    # Perform the integration using the 4th order Runge-Kutta method
    for i in range(1, len(tout)):
        h = tout[i] - tout[i - 1]

        k1 = odefun(tout[i - 1], y)
        k2 = odefun(tout[i - 1] + h / 2, y + h * k1 / 2)
        k3 = odefun(tout[i - 1] + h / 2, y + h * k2 / 2)
        k4 = odefun(tout[i - 1] + h, y + h * k3)
        y = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        yout[i, :] = y
        
    return np.column_stack((tout, yout))

# Function to calculate satellite motion: position and velocity updates
def satellite_motion(t, y, mu):
    r = np.array(y[:3])  # Position vector r = [r_x, r_y, r_z]
    v = np.array(y[3:])  # Velocity vector v = [v_x, v_y, v_z]
    
    r_norm = np.linalg.norm(r)  # Magnitude of the position vector
    
    a = -mu / r_norm**3 * r  # Acceleration vector
    
    dydt = np.concatenate((v, a))  # Combine velocity and acceleration
    
    return dydt

# Function to generate Gauss-Lobatto quadrature points
def gauss_lobatto_points(n, a, b):
    """Generate Gauss-Lobatto points for the interval [a, b]."""
    points = [-np.cos(np.pi * i / (n - 1)) for i in range(n)]
    scaled_points = 0.5 * (b - a) * (np.array(points) + 1) + a
    return scaled_points

def main():
    tspan = [0, 3000]  # Time range from 0 to 3600 seconds
    r0 = np.array([-3.31458372e+03, 2.80902481e+03, -5.40046222e+03])  # Initial position vector
    v0 = np.array([-3.42701791e+00, -6.62341508e+00, -1.34238849e+00])  # Initial velocity vector
    y0 = np.concatenate((r0, v0))  # Initial state vector
    mu = 398600  # Standard gravitational parameter (example for Earth)
    
    # Calculate Gauss-Lobatto points for the given time range
    n_points = 32  # Number of Gauss-Lobatto points
    gauss_lobatto_tspan = gauss_lobatto_points(n_points, tspan[0], tspan[1])

    def func(t, y):
        return satellite_motion(t, y, mu)
    results = rk4(func,gauss_lobatto_tspan, y0)


    

    # Print results in organized format
    print("Results (Time | Position X, Y, Z | Velocity X, Y, Z):")
    print(f"{'Time':^8} {'PosX':^12} {'PosY':^12} {'PosZ':^12} {'VelX':^12} {'VelY':^12} {'VelZ':^12}")
    print("-" * 80)
    
    for i in range(len(results)):
        print(f"{results[i][0]:^8.2f} {results[i][1]:^12.2f} {results[i][2]:^12.2f} {results[i][3]:^12.2f} {results[i][4]:^12.2f} {results[i][5]:^12.2f} {results[i][6]:^12.2f}")

if __name__ == "__main__":
    main()
