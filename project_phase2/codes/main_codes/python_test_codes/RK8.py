import numpy as np
from numpy.polynomial.legendre import leggauss

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
    # Get the Gauss-Lobatto points and weights (Legendre-Gauss-Lobatto points)
    x, _ = leggauss(n)
    
    # Scale points to interval [a, b]
    return 0.5 * ((b - a) * x + (b + a))

# Define the Butcher table for DP8 method
Butcher_table_DP8 = {
    'c': [0, 1/18, 1/12, 1/8, 5/16, 3/8, 59/400, 93/200, 5490023248/9719169821, 13/20, 1201146811/1299019798, 1, 1],
    'a': [
        [],
        [1/18],
        [1/48, 1/16],
        [1/32, 0, 3/32],
        [5/16, 0, -75/64, 75/64],
        [3/80, 0, 0, 3/16, 3/20],
        [29443841/614563906, 0, 0, 77736538/692538347, -28693883/1125000000, 23124283/1800000000],
        [16016141/946692911, 0, 0, 61564180/158732637, 22789713/633445777, 545815736/2771057229, -180193667/1043307555],
        [39632708/573591083, 0, 0, -433636366/683701615, -421739975/2616292301, 100302831/723423059, 790204164/839813087, 800635310/3783071287],
        [246121993/1340847787, 0, 0, -37695042795/15268766246, -309121744/1061227803, -12992083/490766935, 6005943493/2108947869, 393006217/1396673457, 123872331/1001029789],
        [-1028468189/846180014, 0, 0, 8478235783/508512852, 1311729495/1432422823, -10304129995/1701304382, -48777925059/3047939560, 15336726248/1032824649, -45442868181/3398467696, 3065993473/597172653],
        [185892177/718116043, 0, 0, -3185094517/667107341, -477755414/1098053517, -703635378/230739211, 5731566787/1027545527, 5232866602/850066563, -4093664535/808688257, 3962137247/1805957418, 65686358/487910083],
        [403863854/491063109, 0, 0, -5068492393/434740067, -411421997/543043805, 652783627/914296604, 11173962825/925320556, -13158990841/6184727034, 3936647629/1978049680, -160528059/685178525, 248638103/1413531060]
    ],
    'b': [14005451/335480064, 0, 0, 0, 0, -59238493/1068277825, 181606767/758867731, 561292985/797845732, -1041891430/1371343529, 760417239/1151165299, 118820643/751138087, -528747749/2220607170, 1/4],
    'order': 8
}

# RK8 method implementation using Gauss-Lobatto points
def RK8(f, t_gauss_lobatto, Y0 ,h=0.1):
    
    # If only one time point is provided, return the initial condition
    if len(t_gauss_lobatto) == 1:
        t_gauss_lobatto = np.arange( t_gauss_lobatto[0]-0.99*t_gauss_lobatto[0] ,  t_gauss_lobatto[0]+0.05 , 0.1)
        
    # Use Gauss-Lobatto points for the time steps
    tout = np.array(t_gauss_lobatto)
    yout = np.zeros((len(tout), len(Y0)))
    yout[0, :] = Y0
    y = np.array(Y0)
    
    for i in range(1, len(tout)):
        h_step = tout[i] - tout[i - 1]
        k = np.zeros((len(Butcher_table_DP8['c']), len(Y0)))
        
        # Calculate k values using Butcher table coefficients
        for j, cj in enumerate(Butcher_table_DP8['c']):
            if j == 0:
                y_temp = y
            else:
                y_temp = y + h_step * sum(Butcher_table_DP8['a'][j][l] * k[l] for l in range(min(j, len(Butcher_table_DP8['a'][j]))))
            k[j] = f(tout[i - 1] + cj * h_step, y_temp)
        
        # Update state vector y
        y += h_step * sum(Butcher_table_DP8['b'][j] * k[j] for j in range(len(k)))
        yout[i, :] = y

    return np.column_stack((tout, yout))

# Main function to run the simulation
def main():
    TSPAN = [0, 50]  # Time range from 0 to 3600 seconds
    r0 = np.array([-3829.29, 5677.86, -1385.16])  # Initial position vector (example)
    v0 = np.array([-1.69535, -0.63752, 7.33375])   # Initial velocity vector (example)
    y0 = np.concatenate((r0, v0))  # Initial state vector
    mu = 398600  # Standard gravitational parameter (example for Earth)
    
    # Calculate Gauss-Lobatto points for the given time range
    n_points = 5  # Number of Gauss-Lobatto points
    gauss_lobatto_tspan = gauss_lobatto_points(n_points, TSPAN[0], TSPAN[1])

    def func(t, y):
        return satellite_motion(t, y, mu)

    results = RK8(func,[476.54] , y0)
    
    # Print results in organized format
    print("Results (Time | Position X, Y, Z | Velocity X, Y, Z):")
    print(f"{'Time':^8} {'PosX':^12} {'PosY':^12} {'PosZ':^12} {'VelX':^12} {'VelY':^12} {'VelZ':^12}")
    print("-" * 80)
    
    for i in range(len(results)):
        print(f"{results[i][0]:^8.2f} {results[i][1]:^12.2f} {results[i][2]:^12.2f} {results[i][3]:^12.2f} {results[i][4]:^12.2f} {results[i][5]:^12.2f} {results[i][6]:^12.2f}")

if __name__ == "__main__":
    main()
