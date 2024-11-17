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


import csv
import os

def save_results_to_csv(satellite_name, algorithm_name, time_points, position_differences):
    """
    Save the time points and position differences (errors) to a CSV file.
    
    Parameters:
    satellite_name - Name of the satellite
    algorithm_name - Name of the algorithm
    time_points - List of time points in seconds
    position_differences - List of position differences (errors) in kilometers
    """
    filename = f"{satellite_name}_{algorithm_name}_results.csv"
    
    # Ensure directory exists
    os.makedirs('results', exist_ok=True)
    
    filepath = os.path.join('results', filename)
    
    # Write to CSV
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time (s)", "Error (km)"])
        for time_point, error in zip(time_points, position_differences):
            writer.writerow([time_point, error])
    
    print(f"Results saved to {filepath}")

def compare_algorithms(r0, v0, algorithm_name, algorithm_module, func_name, tle_data, year, month, day, hour, minute, second, A, m, C_D=2.2, **kwargs):
    start_time = time.time()
    
    total_time = 1000000
    print(f"Total propagation time for {tle_data[0][0]}: {total_time} seconds")
    min1 = 100000000
    
    if total_time < min1:
        min1 = total_time
        
    total_time = min1
    
    num_segments = 16
    segment_time = total_time / num_segments
    n_points = 32
    y0 = np.concatenate((r0, v0))
    
    position_differences = []
    time_points = []
    
    for i in range(num_segments):
        t_start = i * segment_time
        t_end = t_start + segment_time
        gauss_lobatto_tspan = gauss_lobatto_points(n_points, t_start, t_end)
        
        # Handle the special cases for algorithms that need specific arguments
        if algorithm_name in ["ODE45", "ODE113"]:
            # Pass additional parameters needed for ODE45 and ODE113
            results = getattr(algorithm_module, func_name)(
                lambda t, y: a_c_func(t, y, A, m, C_D), 
                gauss_lobatto_tspan, y0, kwargs.get('tol', 1e-6), kwargs.get('hmax', 0.1), kwargs.get('hmin', 1e-6)
            )
        else:
            # For other algorithms like RK4, RK8, etc.
            results = getattr(algorithm_module, func_name)(
                lambda t, y: a_c_func(t, y, A, m, C_D), gauss_lobatto_tspan, y0, **kwargs
            )
        
        jd, fr = jday(year, month, day, hour, minute, second)
        satellite = Satrec.twoline2rv(tle_data[0][1], tle_data[0][2])
        
        for idx, t_point in enumerate(gauss_lobatto_tspan):
            e, r_sgp4, _ = satellite.sgp4(jd, fr + t_point / 86400.0)
            
            if e == 0:
                current_position = results[idx][1:4]
                position_difference = np.linalg.norm(np.array(current_position) - np.array(r_sgp4))
                position_differences.append(position_difference)
                time_points.append(t_point)
                print(f"Time {t_point}: Position difference = {position_difference} km")
            else:
                print(f"SGP4 error at time {t_point}: {e}")
                position_differences.append(float('nan'))
                time_points.append(t_point)
        
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
    
    # Save the results to CSV
    save_results_to_csv(tle_data[0][0], algorithm_name, time_points, position_differences)



# Updated print_comparison_summary function to include total_time
def print_comparison_summary():
    print("\n*************** Average Position Difference for All Satellites ***************")
    print(f"{'Satellite':<15}{'Algorithm':<10}{'Avg Position Difference (km)':<30}{'Execution Time (s)':<20}{'Total Time (s)':<15}")
    print("-" * 90)
    for result in comparison_results:
        print(f"{result['satellite']:<15}{result['algorithm']:<10}{result['avg_position_difference']:<30.6f}{result['execution_time']:<20.6f}{result['total_time']:<15.6f}")





# The rest of the code remains the same, including the call to the compare_algorithms function for each algorithm.

# Sample TLE data
tle_data = [
    ("STARLINK-1341",
     "1 45534U 19060A   24291.03059268  .00000589  00000-0  58423-4 0  9996",
     "2 45534  53.0541 230.0736 0001515  90.2860 269.8302 15.06388131247839"),

    ("IRIDIUM 33 DEB",
     "1 24946U 97051C   24246.56685503  .00000877  00000-0  30433-3 0  9991",
     "2 24946  86.3813 263.6971 0005980 306.8788  53.1859 14.34480218411509"),

    ("QIANFAN-4",
     "1 60382U 24140D   24290.81465998  .00001664  00000-0  76079-3 0  9996",
     "2 60382  88.9933 342.9506 0013083 358.1907   1.9234 14.21024776 10204"),

    ("SKYNET 4C",
     "1 20776U 90079A   24273.86593681  .00000122  00000-0  00000-0 0  9996",
     "2 20776  13.5422 354.0676 0003521 217.5807 142.6885  1.00279641124611"),

    ("ASBM-2",
     "1 60423U 24143B   24286.24012071  .00000053  00000-0  00000-0 0  9996",
     "2 60423  62.3237  68.6665 5330361 267.8786  33.3421  1.50421044  2382"),
]

current_time_utc = datetime.datetime.utcnow()
year, month, day, hour, minute, second = (
    current_time_utc.year, current_time_utc.month, current_time_utc.day, 
    current_time_utc.hour, current_time_utc.minute, current_time_utc.second)

# Iterate over all satellites and compare algorithms for each
for sat_data in tle_data:
    results = compute_position_velocity_and_geodetic([sat_data], year, month, day, hour, minute, second)
    result = results[0]
    if 'error' not in result:
        r0 = np.array([result['position']['x'], result['position']['y'], result['position']['z']])
        v0 = np.array([result['velocity']['vx'], result['velocity']['vy'], result['velocity']['vz']])
        
        A, m, C_D = get_satellite_params(sat_data[0])

        if A is not None and m is not None:
            compare_algorithms(r0, v0, "RK4", RK4, "rk4", [sat_data], year, month, day, hour, minute, second, 3.9e-6, 260, C_D)
            compare_algorithms(r0, v0, "RK8", RK8, "RK8", [sat_data], year, month, day, hour, minute, second, A, m, C_D)
            compare_algorithms(r0, v0, "ODE45", ODE45, "ode45", [sat_data], year, month, day, hour, minute, second, A, m, C_D)
            compare_algorithms(r0, v0, "ODE78", ODE78, "ode78", [sat_data], year, month, day, hour, minute, second, A, m, C_D)
            compare_algorithms(r0, v0, "ODE113", ODE113, "ODE113", [sat_data], year, month, day, hour, minute, second, A, m, C_D)
        else:
            print(f"Satellite parameters not found for {sat_data[0]}.")
    else:
        print(f"Error computing initial position and velocity for satellite {sat_data[0]}: {result['error']}")

# Print summary
print_comparison_summary()
