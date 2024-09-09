import numpy as np

def MPCI(func, tspan, y0, h, N, tol=1e-10, max_iter=100):
    """
    Modified Picard-Chebyshev Iteration method to solve ODEs.
    
    Parameters:
    func  - function defining the ODE (dy/dt = func(t, y))
    tspan - tuple (t0, tf) specifying the time range
    y0    - initial condition
    h     - step size
    N     - number of Chebyshev nodes
    tol   - tolerance for convergence
    max_iter - maximum number of iterations
    
    Returns:
    results - matrix of time points and solution values
    """
    t0, tf = tspan
    t_values = np.arange(t0, tf + h, h)
    M = N + 1
    y_values = np.zeros((len(t_values), len(y0)))
    y_values[0] = y0
    tau = np.cos(np.linspace(0, np.pi, M))

    # Initialize variables for iteration
    Xn = np.zeros(M * len(y0))
    Xo = np.zeros_like(Xn)
    xAdd = np.zeros_like(Xn)
    temp = 0.0
    
    # Precompute Chebyshev coefficients
    Im = MPCI_CoeffsI(N, M)
    
    for i in range(1, len(t_values)):
        t = t_values[i-1]
        y = y_values[i-1]
        x0 = y
        for iteration in range(max_iter):
            # Update Xn based on the previous step and current estimate
            for node in range(M):
                tau_val = tau[node]
                tn = t + h * (tau_val + 1) / 2
                xAdd[node * len(y0):(node + 1) * len(y0)] = func(tn, y)
                
            # Update using error and convergence check
            errorAndUpdate(M, h, len(y0), x0, Xo, Xn, xAdd, temp)
            
            # Check for convergence
            if np.linalg.norm(Xn - Xo) < tol:
                break
            Xo = np.copy(Xn)
        
        # Store the result at the current time step
        y_values[i] = Xn[:len(y0)]
    
    results = np.column_stack((t_values, y_values))  # Combine time and state values
    return results

def MPCI_CoeffsI(N, M):
    W = np.zeros((M, M))
    T = np.zeros((N + 1, M))
    TT = np.zeros((M, N + 1))
    T2 = np.zeros((N + 2, M))
    T2Z = np.zeros((N + 2, M))
    Z = np.zeros((M + 2, M))
    V = np.zeros((N + 1, N + 2))
    I_N = np.zeros((N + 2, N + 2))
    tau = np.zeros(M)
    
    for i in range(M):
        tau[i] = np.cos(i * np.pi / N + np.pi)
    
    # BUILD W MATRIX (symmetric)
    W[0, 0] = 0.5
    for i in range(1, M - 1):
        W[i, i] = 1.0
    W[M - 1, M - 1] = 0.5
    
    # BUILD T MATRIX (symmetric)
    for j in range(M):
        for i in range(N + 1):
            T[i, j] = np.cos(i * np.arccos(tau[j]))

    # BUILD TT MATRIX (symmetric)
    for j in range(N + 1):
        for i in range(M):
            TT[i, j] = np.cos(j * np.arccos(tau[i]))

    # BUILD T2 MATRIX (symmetric)
    for j in range(M):
        for i in range(N + 2):
            T2[i, j] = np.cos(i * np.arccos(tau[j]))

    # BUILD T2Z MATRIX (symmetric)
    for j in range(M):
        for i in range(N + 2):
            T2Z[i, j] = np.cos(i * np.arccos(tau[j])) - (-1) ** (i + 2)

    # BUILD Z MATRIX (symmetric)
    for j in range(M):
        for i in range(N + 2):
            Z[i, j] = (-1) ** (i + 2)

    # BUILD V MATRIX (symmetric)
    vElem = 1.0 / N
    V[0, 0] = vElem
    V[N, N] = vElem
    for i in range(1, N):
        V[i, i] = 2.0 * vElem
    
    I_N[0, 1] = 1.0
    I_N[1, 0] = 0.25
    I_N[1, 2] = 0.25
    
    for ii in range(2, N + 1):
        I_N[ii, ii - 1] = -0.5 / (ii - 1)
        if ii < N + 1:
            I_N[ii, ii + 1] = 0.5 / ii
    
    # Building Cx & Ca matrices
    WT = np.matmul(W, TT)
    WTV = np.matmul(WT, V)
    ITZ = np.matmul(I_N, T2Z)
    
    Im = np.matmul(WTV, ITZ)
    
    return Im

def errorAndUpdate(MM, timeSub, Nstates2, x0, Xo, Xn, xAdd, temp):
    for node in range(MM):
        for state in range(Nstates2):
            indx = node * Nstates2 + state
            Xn[indx] = x0[state] + timeSub * xAdd[indx]
            Err = abs(Xn[indx] - Xo[indx]) / max(1.0, abs(Xo[indx]))
            
            if state == 0:  # Initialize temp with the first state's error
                temp = Err
            if Err > temp:
                temp = Err
            Xo[indx] = Xn[indx]

def satellite_motion(t, y, mu):
    r = np.array(y[:3])
    v = np.array(y[3:])
    
    r_norm = np.linalg.norm(r)
    
    a = -mu / r_norm**3 * r
    
    dydt = np.concatenate((v, a))
    
    return dydt

def main():
    tspan = (0, 3600)
    r0 = np.array([-3829.29, 5677.86, -1385.16])
    v0 = np.array([-1.69535, -0.63752, 7.33375])
    y0 = np.concatenate((r0, v0))
    mu = 398600
    h = 0.1
    N = 40

    def func(t, y):
        return satellite_motion(t, y, mu)

    results = MPCI(func, tspan, y0, h, N, tol=1e-10)
    
    print("Results (Time | Position X, Y, Z | Velocity X, Y, Z):")
    print(f"{'Time':^8} {'PosX':^12} {'PosY':^12} {'PosZ':^12} {'VelX':^12} {'VelY':^12} {'VelZ':^12}")
    print("-" * 80)
    for row in results:
        print(f"{row[0]:^8.2f} {row[1]:^12.2f} {row[2]:^12.2f} {row[3]:^12.2f} {row[4]:^12.2f} {row[5]:^12.2f} {row[6]:^12.2f}")

if __name__ == "__main__":
    main()
