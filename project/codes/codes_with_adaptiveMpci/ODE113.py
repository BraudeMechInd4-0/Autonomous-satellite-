import numpy as np

def rk4_step(f, t, y, h):
    """Performs a single step of the classic 4th-order Runge-Kutta method."""
    k1 = h * f(t, y)
    k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
    k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def adams_bashforth_moulton_adaptive(f, tspan, y0, tol, hmax=1.0,hmin=1e-6):
    """
    Adams-Bashforth-Moulton method with adaptive step size to solve ODEs.
    
    Parameters:
    f      - function defining the ODE (dy/dx = f(t, y))
    tspan  - tuple (t0, tf) specifying the time range
    y0     - initial condition
    tol    - tolerance for adaptive step size
    hmin   - minimum allowable step size
    hmax   - maximum allowable step size
    
    Returns:
    t_values - array of time points
    y_values - array of solution values corresponding to time points
    """
    t0, tf = tspan
    h = (tf - t0) / 1000  # Initial step size estimate
    t_values_list = [t0]
    y_values_list = [y0]

    # Initial conditions for the first few steps using RK4
    t = t0
    y = y0

    # Compute initial points using RK4
    for _ in range(3):
        y_next = rk4_step(f, t, y, h)
        t += h
        t_values_list.append(t)
        y_values_list.append(y_next)
        y = y_next

    # Adams-Bashforth-Moulton method with adaptive step size
    while t < tf:
        if len(t_values_list) < 4:
            y_pred = rk4_step(f, t, y, h)
        else:
            # Adams-Bashforth predictor
            y_pred = y_values_list[-1] + h/24 * (
                55*f(t_values_list[-1], y_values_list[-1])
                - 59*f(t_values_list[-2], y_values_list[-2])
                + 37*f(t_values_list[-3], y_values_list[-3])
                - 9*f(t_values_list[-4], y_values_list[-4])
            )

        # Adams-Moulton corrector
        t_next = t + h
        y_correct = y_values_list[-1] + h/24 * (
            9*f(t_next, y_pred)
            + 19*f(t_values_list[-1], y_values_list[-1])
            - 5*f(t_values_list[-2], y_values_list[-2])
            + f(t_values_list[-3], y_values_list[-3])
        )

        # Error estimation
        error = np.abs(y_correct - y_pred)

        # Step size adjustment
        if np.any(error > tol):
            h *= 0.9 * (tol / np.max(error)) ** 0.25
            if h < hmin:
                print("Warning: Step size too small. Ending integration.")
                break
        else:
            t_values_list.append(t_next)
            y_values_list.append(y_correct)
            t = t_next  # Update time
            h = min(h * 1.5, hmax)  # Increase step size for efficiency

        # Check if we're close to the final time to avoid getting stuck
        if tf - t < hmin:
            print("Warning: Close to final time. Ending integration.")
            break
    results = np.column_stack((t_values_list, y_values_list))  # Combine time and state values
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
    tol = 1e-4  # Tolerance for adaptive step size

    def func(t, y):
        return satellite_motion(t, y, mu)

    results = adams_bashforth_moulton_adaptive(func, tspan, y0, tol)
    
    print("Results (Time | Position X, Y, Z | Velocity X, Y, Z):")
    print(f"{'Time':^8} {'PosX':^12} {'PosY':^12} {'PosZ':^12} {'VelX':^12} {'VelY':^12} {'VelZ':^12}")
    print("-" * 80)
    for row in results:
        print(f"{row[0]:^8.2f} {row[1]:^12.2f} {row[2]:^12.2f} {row[3]:^12.2f} {row[4]:^12.2f} {row[5]:^12.2f} {row[6]:^12.2f}")

if __name__ == "__main__":
    main()
