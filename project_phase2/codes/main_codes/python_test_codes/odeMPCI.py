import numpy as np
from scipy.integrate import odeint

def odeMPCI(ODEFUN, TSPAN, X0, AbsTol=1e-6, RelTol=1e-6, N=32, xtinit=None):
    """
    Bai's MPCI integration method, as described in the referenced papers.
    
    Parameters:
    ODEFUN - The function to integrate (the f in the equation dx/dt = f(x,t))
    TSPAN  - The time span of the integration. Can be [Tstart, Tend] or a vector of timepoints to be evaluated
    X0     - Initial conditions (should be given as x(t = 0))
    AbsTol - Absolute tolerance
    RelTol - Relative tolerance
    N      - Number of timepoints to be sampled. Optional if TSPAN is a vector longer than 2, mandatory if it is [Tstart, Tend]
    xtinit - An initial guess. For astrodynamics, it is customary to use the solution for the two-body problem.
    
    Returns:
    tout   - A list of time points
    xout   - Values of the function approximation on these points
    error  - Error code: 0 if all went well, -1 if failed to converge
    """
    
    error = 0
    if len(TSPAN) > 2:
        N = len(TSPAN) - 1
    else:
        N = N - 1  # Because it's 0 - N in all places, so if N is 32 and we don't deduct one, we get 33 points.

    tau = np.cos(np.arange(N + 1) * np.pi / N)  # Gauss-Lobatto nodes
    if xtinit is None:
        xtinit = np.tile(X0, (len(tau), 1))  # Initial guess

    X0 = np.vstack((X0, np.zeros((N, len(X0)))))

    xold = xtinit  # Initial guess
    om2 = (TSPAN[-1] - TSPAN[0]) / 2
    om1 = (TSPAN[-1] + TSPAN[0]) / 2

    # Create the vectors and matrices
    W = np.eye(len(tau))
    W[0, 0] = 0.5
    W[-1, -1] = 0.5

    # Corrected T matrix construction
    T = np.cos(np.outer(np.arange(N + 1), np.arccos(tau)))  # Shape: (N+1, N+1)

    Tm1 = np.cos(np.arange(N + 1) * np.pi)  # acos(-1) = pi
    L = np.vstack((Tm1, np.zeros((N, N + 1))))
    s_ = 1 / np.arange(4, 2 * N + 1, 2)
    S_3 = np.zeros((N + 1, N + 1))
    S_3[1:, 1:] = np.diag([-0.5] + list(-s_[:-1]), 1)
    S_2 = np.diag([1] + list(s_), -1)
    S_1 = S_2 + S_3
    S = S_1  # Use the full S matrix, shape: (N+1, N+1)

    # Corrected A matrix calculation
    A = np.linalg.inv(T.T @ W @ T) @ T.T @ W

    # Get the function g using Picard iterations
    T = np.cos(np.outer(np.arange(N + 1), np.arccos(tau)))
    eAbs = np.inf
    eRel = np.inf
    i = 0
    while (eAbs > AbsTol or eRel > RelTol) and i < 5000:
        F = ODEFUN(om2 * tau + om1, xold)  # The VMPCM uses F = ode(input{:}).*omega2
        P1 = om2 * (np.eye(N + 1) - L) @ S
        bi = X0 + P1 @ A @ F
        xnew = T @ bi
        eAbs = np.max(np.abs(xnew - xold))
        with np.errstate(divide='ignore', invalid='ignore'):
            eRel = np.max(np.abs((xnew - xold) / np.maximum(np.abs(xold), 1e-12)))
        xold = xnew
        i += 1

    if i >= 5000:
        error = -1
        tout = 0
        xout = 0
        return tout, xout, error

    # Evaluate at the required t's
    if len(TSPAN) != 2:
        tau = -(TSPAN[-1] - TSPAN[0]) / (TSPAN[-1] + TSPAN[0]) + 2 * TSPAN / (TSPAN[-1] - TSPAN[0])
        tout = TSPAN
    else:
        tout = TSPAN[0] + (tau + 1) * (TSPAN[-1] - TSPAN[0]) / 2

    T = np.cos(np.outer(np.arange(N + 1), np.arccos(tau)))
    xout = np.zeros((len(tau), 6))
    for i in range(len(tau)):
        xout[i, :] = T[i, :] @ bi

    return tout, xout, error

def orbit_eq(r, t, mu):
    """
    Definition of the orbit equation.
    """
    dr = np.zeros_like(r)
    le = np.linalg.norm(r[:3])**3
    if le == 0:
        le = 1e-12  # Avoid division by zero
    dr[:3] = r[3:6]
    dr[3:6] = -mu * r[:3] / le
    return dr

def main():
    
    # Test case
    mu = 398600.4418  # Earth's gravitational parameter (km^3/s^2)
    r0 = np.array([-3.31458372e+03, 2.80902481e+03, -5.40046222e+03, -3.42701791e+00, -6.62341508e+00, -1.34238849e+00])  # Initial position vector
    tspan = np.linspace(0, 2000, 50)  # Time span from 0 to 2000 seconds with 50 points
    
    # Using odeMPCI
    tout, xout, error = odeMPCI(lambda t, r: orbit_eq(r, t, mu), tspan, r0)
    
    # Combine time and state vectors for printing
    result = np.hstack((tout.reshape(-1, 1), xout))
    
    # Print results in a formatted table
    print("Results (Time | Position X, Y, Z | Velocity X, Y, Z):")
    print(f"{'Time':^8} {'PosX':^12} {'PosY':^12} {'PosZ':^12} {'VelX':^12} {'VelY':^12} {'VelZ':^12}")
    print("-" * 80)
    
    for row in result:
        print(f"{row[0]:^8.2f} {row[1]:^12.2f} {row[2]:^12.2f} {row[3]:^12.2f} {row[4]:^12.2f} {row[5]:^12.2f} {row[6]:^12.2f}")
    
        
if __name__ == "__main__":
    main()