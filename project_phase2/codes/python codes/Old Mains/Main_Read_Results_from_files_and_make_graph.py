import datetime
import math
import numpy as np
from sgp4.api import Satrec, jday
import RK4
import RK8
import ODE45
import ODE78
import ODE113
import time
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
import pandas as pd  # Import pandas to read the CSV file

# Constants
EARTH_RADIUS_KM = 6378.137  # WGS84 Earth radius at the equator in kilometers
comparison_results = []

def a_c_func(t, y, A, m, C_D=2.2):
    """
    Computes the time derivative of the state vector (position and velocity) including J2 perturbation and atmospheric drag.
    
    Parameters:
    t - Time (not used in this simple model)
    y - State vector [x, y, z, v_x, v_y, v_z]
    A - Cross-sectional area of the satellite (in square meters)
    m - Mass of the satellite (in kilograms)
    C_D - Drag coefficient (dimensionless)
    
    Returns:
    dy - Time derivative of the state vector [v_x, v_y, v_z, a_x, a_y, a_z]
    """
    # Extract position and velocity from the state vector
    r = y[:3]  # Position vector [x, y, z]
    v = y[3:]  # Velocity vector [v_x, v_y, v_z]

    # Constants for J2 correction
    J2 = 1.08263e-3  # Earth's J2 coefficient (dimensionless)
    R_E = 6378.137  # Earth's radius in kilometers
    mu = 398600.4418  # Earth's gravitational parameter (km^3/s^2)
    
    r_norm = np.linalg.norm(r)  # Magnitude of the position vector
    x, y_pos, z = r  # Decompose the position vector into x, y, z components
    
    # J2 perturbation correction
    factor = (3 / 2) * J2 * (R_E ** 2) / (r_norm ** 5)
    z2_r2 = (z / r_norm) ** 2  # (z / r)^2 term
    
    # J2 acceleration components
    a_c_x = factor * x * (1 - 5 * z2_r2)
    a_c_y = factor * y_pos * (1 - 5 * z2_r2)
    a_c_z = factor * z * (3 - 5 * z2_r2)
    
    # Total J2 correction acceleration
    a_c = np.array([a_c_x, a_c_y, a_c_z])
    
    # Atmospheric drag calculation
    altitude = r_norm - R_E  # Altitude in kilometers
    rho = atmospheric_density(altitude)  # Atmospheric density at the given altitude
    
    v_rel = v  # Assuming atmosphere velocity is negligible, so v_rel ≈ v
    
    # Drag acceleration: a_d = -1/2 * (C_D * A / m) * rho * |v_rel| * v_rel
    v_rel_norm = np.linalg.norm(v_rel)
    drag_factor = -0.5 * C_D * A / m * rho
    a_d = drag_factor * v_rel_norm * v_rel
    
    # Gravitational acceleration: a_g = -mu / r_norm^3 * r
    a_g = (-mu / r_norm**3) * r
    
    # Total acceleration (gravitational + J2 + drag)
    total_acceleration = a_g + a_c + a_d
    
    # The derivative of position is the velocity
    drdt = v
    
    # The derivative of velocity is the acceleration
    dvdt = total_acceleration
    
    # Combine the derivatives into a single array
    dy = np.hstack((drdt, dvdt))
    
    return dy



def atmospheric_density(altitude):
    """
    Returns the atmospheric density based on the altitude using an exponential model.
    
    Parameters:
    altitude - Altitude above Earth's surface in kilometers
    
    Returns:
    rho - Atmospheric density in kg/m^3
    """
    # Table values for base altitude, nominal density, and scale height (from the provided table)
    altitude_data = [
        (0, 1.225, 7.249), (25, 3.899e-2, 6.349), (30, 1.774e-2, 6.682),
        (40, 3.972e-3, 7.554), (50, 1.057e-3, 8.382), (60, 3.206e-4, 7.714),
        (70, 8.77e-5, 6.549), (80, 1.905e-5, 5.799), (90, 3.396e-6, 5.382),
        (100, 5.297e-7, 5.877), (110, 9.661e-8, 7.263), (120, 2.438e-8, 9.473),
        (130, 8.484e-9, 12.636), (140, 3.845e-9, 16.149), (150, 2.070e-9, 22.523),
        (180, 5.464e-10, 29.740), (200, 2.789e-10, 37.105), (250, 7.248e-11, 45.546),
        (300, 2.418e-11, 53.628), (350, 9.518e-12, 53.298), (400, 3.725e-12, 58.515),
        (450, 1.585e-12, 60.828), (500, 6.967e-13, 63.822), (600, 1.454e-13, 71.835),
        (700, 3.614e-14, 88.667), (800, 1.170e-14, 124.64), (900, 5.245e-15, 181.05),
        (1000, 3.019e-15, 268.00)
    ]
    
    for i in range(len(altitude_data) - 1):
        h0, rho0, H = altitude_data[i]
        h1, rho1, _ = altitude_data[i + 1]
        
        if h0 <= altitude < h1:
            # Use exponential model to calculate density
            return rho0 * np.exp(-(altitude - h0) / H)
    
    # If altitude is above 1000 km, return the smallest density value
    return altitude_data[-1][1]

def get_satellite_params(sat_name):
    """
    Returns the cross-sectional area, mass, and drag coefficient for a given satellite.
    
    Parameters:
    sat_name - Name of the satellite (as a string)
    
    Returns:
    A - Cross-sectional area in square meters
    m - Mass in kilograms
    C_D - Drag coefficient (assumed)
    """
    satellite_data = {
        "STARLINK-1341": (3.9, 260, 2.2),
        "IRIDIUM 33 DEB": (0.7, 77, 2.2),
        "QIANFAN-4": (4.0, 260, 2.2),
        "SKYNET 4C": (10.0, 1250, 2.2),
        "ASBM-2": (12.0, 2000, 2.2),
    }
    
    return satellite_data.get(sat_name, (None, None, None))


def compute_orbital_period_in_seconds(tle_data):
    # Extract semi-major axis (a) in km from TLE data
    # You may need to convert or extract semi-major axis using SGP4 or a similar method.
    satellite = Satrec.twoline2rv(tle_data[0][1], tle_data[0][2])
    a = (satellite.a * 6378.137)  # Earth radius in km to get semi-major axis in km
    
    # Use Kepler's Third Law to compute orbital period in seconds
    # T = 2 * pi * sqrt(a^3 / GM)
    GM = 398600.4418  # Earth's gravitational constant in km^3/s^2
    orbital_period = 2 * np.pi * np.sqrt(a**3 / GM)
    
    return orbital_period  # This will be in seconds

def satellite_motion_only_gravity(t, y, mu):
    r = np.array(y[:3])  # Position vector r = [r_x, r_y, r_z]
    v = np.array(y[3:])  # Velocity vector v = [v_x, v_y, v_z]
    r_norm = np.linalg.norm(r)  # Magnitude of the position vector
    a = -mu / r_norm**3 * r  # Acceleration vector
    dydt = np.concatenate((v, a))  # Combine velocity and acceleration
    return dydt


def gauss_lobatto_points(n, a, b):
    points = [-np.cos(np.pi * i / (n - 1)) for i in range(n)]
    scaled_points = 0.5 * (b - a) * (np.array(points) + 1) + a
    return scaled_points

def eci_to_geodetic(x, y, z):
    r = math.sqrt(x**2 + y**2 + z**2)
    lon = math.degrees(math.atan2(y, x))
    lat = math.degrees(math.atan2(z, math.sqrt(x**2 + y**2)))
    alt = r - EARTH_RADIUS_KM
    return lat, lon, alt

def compute_position_velocity_and_geodetic(tle_data, year, month, day, hour, minute, second):
    jd, fr = jday(year, month, day, hour, minute, second)
    results = []
    
    for object_name, tle_line1, tle_line2 in tle_data:
        satellite = Satrec.twoline2rv(tle_line1, tle_line2)
        e, r, v = satellite.sgp4(jd, fr)
        if e == 0:
            lat, lon, alt = eci_to_geodetic(r[0], r[1], r[2])
            results.append({
                'object_name': object_name,
                'tle_line1': tle_line1,
                'tle_line2': tle_line2,
                'position': {'x': r[0], 'y': r[1], 'z': r[2]},
                'velocity': {'vx': v[0], 'vy': v[1], 'vz': v[2]},
                'geodetic': {'latitude': lat, 'longitude': lon, 'altitude': alt}
            })
        else:
            results.append({
                'object_name': object_name,
                'tle_line1': tle_line1,
                'tle_line2': tle_line2,
                'error': e
            })
    return results


def compare_algorithms(r0, v0, algorithm_name, algorithm_module, func_name, tle_data, year, month, day, hour, minute, second, A, m, C_D=2.2, **kwargs):
    start_time = time.time()

    # Total time to propagate (seconds)
    total_time = 1000000
    num_segments = 16
    segment_time = total_time / num_segments
    n_points = 32  # Number of Gauss-Lobatto points
    y0 = np.concatenate((r0, v0))  # Initial state vector: position + velocity
    
    time_points = []  # List to store time points
    position_differences = []  # List to store position differences

    # Ensure tol is passed correctly for ODE45 or ODE113
    tol = kwargs.get('tol', 1e-6)  # Default to 1e-6 if not passed
    hmax = kwargs.get('hmax', 0.1)
    hmin = kwargs.get('hmin', 1e-6)

    for i in range(num_segments):
        t_start = i * segment_time
        t_end = t_start + segment_time
        gauss_lobatto_tspan = gauss_lobatto_points(n_points, t_start, t_end)
        
        # Call the appropriate algorithm function for each Gauss-Lobatto point
        if algorithm_name == "APCI":
            results = algorithm_module(lambda t, y: a_c_func(t, y, A, m, C_D), gauss_lobatto_tspan, y0, **kwargs)
        elif algorithm_name == "ODE45" or algorithm_name == "ODE113":  # Specific args for ODE45/ODE113
            results = getattr(algorithm_module, func_name)(
                lambda t, y: a_c_func(t, y, A, m, C_D), 
                gauss_lobatto_tspan, 
                y0, 
                tol=tol, 
                hmax=hmax, 
                hmin=hmin
            )
        else:  # General case for other algorithms (RK4, RK8, ODE78)
            results = getattr(algorithm_module, func_name)(
                lambda t, y: a_c_func(t, y, A, m, C_D),
                gauss_lobatto_tspan,
                y0,
                **kwargs
            )

        jd, fr = jday(year, month, day, hour, minute, second)
        satellite = Satrec.twoline2rv(tle_data[0][1], tle_data[0][2])

        for idx, t_point in enumerate(gauss_lobatto_tspan):
            e, r_sgp4, _ = satellite.sgp4(jd, fr + t_point / 86400.0)

            if e == 0:
                current_position = results[idx][1:4]
                position_difference = np.linalg.norm(np.array(current_position) - np.array(r_sgp4))
                time_points.append(t_point)
                position_differences.append(position_difference)
                print(f"Time {t_point}: Position difference = {position_difference} km")
            else:
                print(f"SGP4 error at time {t_point}: {e}")
                time_points.append(t_point)
                position_differences.append(float('nan'))

        y0 = np.concatenate((results[-1][1:4], results[-1][4:7]))
    
    avg_position_difference = np.nanmean(position_differences)
    elapsed_time = time.time() - start_time

    comparison_results.append({
        'satellite': tle_data[0][0],
        'algorithm': algorithm_name,
        'avg_position_difference': avg_position_difference,
        'execution_time': elapsed_time,
        'total_time': total_time
    })

    return time_points, position_differences

import os

# Function to load results from CSV files located in the 'results' folder
def load_algorithm_results(satellite_name, algorithm_name):
    folder_path = 'results'  # Folder where the result files are stored
    file_name = f'{satellite_name}_{algorithm_name}_results.csv'
    file_path = os.path.join(folder_path, file_name)  # Full path to the file
    df = pd.read_csv(file_path)
    return df['Time (s)'].values, df['Error (km)'].values  # Assuming columns 'Time (s)' and 'Error (km)'

# Updated function to plot results for all algorithms
def plot_all_algorithms(satellite_name):
    algorithms = ["RK4", "RK8", "ODE45", "ODE78", "ODE113", "APC"]  # List of algorithm names
    
    plt.figure(figsize=(10, 6))  # Set figure size
    
    # Plot results of each algorithm
    for algorithm_name in algorithms:
        time_points, position_differences = load_algorithm_results(satellite_name, algorithm_name)
        plt.plot(time_points, position_differences, label=f'{algorithm_name}')
    
    plt.xscale('log')  # Set x-axis to logarithmic scale
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.xlabel('Time (s)')
    plt.ylabel('Position Difference (km)')
    plt.title(f'Position Difference vs. Time for {satellite_name}')
    plt.grid(True)
    plt.legend()
    plt.show()

# Example usage
satellite_name = "QIANFAN-4"  # Replace with the desired satellite name
plot_all_algorithms(satellite_name)
