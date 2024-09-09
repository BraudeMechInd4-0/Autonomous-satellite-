import numpy as np

def MPCI(func, tspan, y0, h_init=0.1,tol=1e-10, max_iter=100, N_init=20):
    """
    Adaptive Modified Picard-Chebyshev Iteration method to solve ODEs with dynamic node adjustment and
    error feedback.
    
    Parameters:
    func  - function defining the ODE (dy/dt = func(t, y))
    tspan - tuple (t0, tf) specifying the time range
    y0    - initial condition
    tol   - tolerance for convergence
    max_iter - maximum number of iterations
    N_init - initial number of Chebyshev nodes
    h_init - initial time step size
    
    Returns:
    results - matrix of time points and solution values
    """
    t0, tf = tspan
    t_values = [t0]
    y_values = [y0]
    
    h = h_init  # Start with a smaller time step
    N = N_init  # initial number of nodes
    M = N + 1
    tau = np.cos(np.linspace(0, np.pi, M))  # Chebyshev nodes in [-1, 1]

    def adjust_nodes(error):
        """
        Dynamically adjust the number of Chebyshev nodes based on the error.
        """
        if error < tol / 10:
            return max(10, N - 5)
        elif error > tol:
            return min(50, N + 5)
        return N

    while t_values[-1] < tf:
        t = t_values[-1]
        y = y_values[-1]
        prev_y = np.copy(y)
        
        # Iterate to refine the solution
        for iteration in range(max_iter):
            Xn = np.zeros((M, len(y)))  # Chebyshev states
            xAdd = np.zeros_like(Xn)
            
            # Map Chebyshev nodes to actual time
            for node in range(M):
                tau_val = tau[node]
                tn = t + h * (tau_val + 1) / 2  # map Chebyshev nodes to time
                xAdd[node] = func(tn, y)
            
            # Integrate using Chebyshev nodes
            y_new = y + h * np.dot(xAdd.T, (1 / M) * np.ones(M))  # weighted sum of Chebyshev nodes
            
            # Calculate error between iterations
            error = np.linalg.norm(y_new - prev_y)
            prev_y = np.copy(y_new)

            # Adjust number of nodes dynamically based on error
            N = adjust_nodes(error)
            M = N + 1
            tau = np.cos(np.linspace(0, np.pi, M))
            
            if error < tol:
                break  # Converged
        
        t_values.append(t + h)
        y_values.append(y_new)

    results = np.column_stack((t_values, np.array(y_values)))
    return results

def satellite_motion(t, y, mu):
    r = np.array(y[:3])
    v = np.array(y[3:])
    
    r_norm = np.linalg.norm(r)
    
    # Gravitational acceleration using inverse-square law
    a = -mu / r_norm**3 * r
    
    dydt = np.concatenate((v, a))
    return dydt

def main():
    tspan = (0, 3600)  # time span (0 to 3600 seconds)
    r0 = np.array([-3829.29, 5677.86, -1385.16])  # initial position
    v0 = np.array([-1.69535, -0.63752, 7.33375])  # initial velocity
    y0 = np.concatenate((r0, v0))  # initial state (position + velocity)
    mu = 398600  # Earth's gravitational parameter in km^3/s^2

    def func(t, y):
        return satellite_motion(t, y, mu)

    # Solve using adaptive MPCI with a smaller initial time step
    results = MPCI(func, tspan, y0, tol=1e-10, h_init=0.1)

    print("Results (Time | Position X, Y, Z | Velocity X, Y, Z):")
    print(f"{'Time':^8} {'PosX':^12} {'PosY':^12} {'PosZ':^12} {'VelX':^12} {'VelY':^12} {'VelZ':^12}")
    print("-" * 80)
    for row in results:
        print(f"{row[0]:^8.2f} {row[1]:^12.2f} {row[2]:^12.2f} {row[3]:^12.2f} {row[4]:^12.2f} {row[5]:^12.2f} {row[6]:^12.2f}")

if __name__ == "__main__":
    main()
