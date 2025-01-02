import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sgp4.api import Satrec, jday
import ODE78
# Constants
EARTH_RADIUS_KM = 6378.137  # WGS84 Earth radius at the equator in kilometers
comparison_results = []
def eci_to_geodetic(x, y, z):
    """
    Converts Earth-Centered Inertial (ECI) coordinates to geodetic coordinates (latitude, longitude, altitude).
    
    Parameters:
    x, y, z - ECI position coordinates (in kilometers)
    
    Returns:
    lat - Latitude in degrees
    lon - Longitude in degrees
    alt - Altitude above Earth's surface in kilometers
    """
    r = math.sqrt(x**2 + y**2 + z**2)  # Calculate the radial distance
    lon = math.degrees(math.atan2(y, x))  # Longitude is the angle in the xy-plane
    lat = math.degrees(math.atan2(z, math.sqrt(x**2 + y**2)))  # Latitude is the angle from the equator plane
    alt = r - EARTH_RADIUS_KM  # Altitude is the radial distance minus Earth's radius
    return lat, lon, alt

def a_c_func(t, y, A, m, C_D=2.2):
    r = y[:3]  # Position vector [x, y, z]
    v = y[3:]  # Velocity vector [v_x, v_y, v_z]

    J2 = 1.08263e-3  # Earth's J2 coefficient (dimensionless)
    R_E = 6378.137  # Earth's radius in kilometers
    mu = 398600.4418  # Earth's gravitational parameter (km^3/s^2)
    
    r_norm = np.linalg.norm(r)  # Magnitude of the position vector
    x, y_pos, z = r  # Decompose the position vector into x, y, z components
    
    factor = (3 / 2) * J2 * (R_E ** 2) / (r_norm ** 5)
    z2_r2 = (z / r_norm) ** 2  # (z / r)^2 term
    
    # J2 acceleration components
    a_c_x = factor * x * (1 - 5 * z2_r2)
    a_c_y = factor * y_pos * (1 - 5 * z2_r2)
    a_c_z = factor * z * (3 - 5 * z2_r2)
    
    a_c = np.array([a_c_x, a_c_y, a_c_z])
    
    altitude = r_norm - R_E  # Altitude in kilometers
    rho = atmospheric_density(altitude)  # Atmospheric density at the given altitude
    
    v_rel = v  # Assuming atmosphere velocity is negligible, so v_rel ≈ v
    v_rel_norm = np.linalg.norm(v_rel)
    drag_factor = -0.5 * C_D * A / m * rho
    a_d = drag_factor * v_rel_norm * v_rel
    
    a_g = (-mu / r_norm**3) * r
    total_acceleration = a_g + a_c + a_d
    
    drdt = v
    dvdt = total_acceleration
    dy = np.hstack((drdt, dvdt))
    
    return dy

def atmospheric_density(altitude):
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
            return rho0 * np.exp(-(altitude - h0) / H)
    
    return altitude_data[-1][1]

def get_satellite_params(sat_name):
    satellite_data = {
        "STARLINK-1341": (3.9, 260, 2.2),
        "IRIDIUM 33 DEB": (0.7, 77, 2.2),
        "QIANFAN-4": (4.0, 260, 2.2),
        "SKYNET 4C": (10.0, 1250, 2.2),
        "ASBM-2": (12.0, 2000, 2.2),
    }
    
    return satellite_data.get(sat_name, (None, None, None))

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
def satellite_motion(t, y, mu= 398600.4418):
    r = np.array(y[:3])  # Position vector
    v = np.array(y[3:])  # Velocity vector
    
    r_norm = np.linalg.norm(r)  # Magnitude of the position vector
    a = -mu / r_norm**3 * r  # Acceleration vector
    
    dydt = np.concatenate((v, a))  # Combine velocity and acceleration
    return dydt
def compare_csv_with_ode78_and_plot(ax, tle_data, year, month, day, hour, minute, second, csv_files):
    """
    Compares multiple algorithms' position data from CSV files with positions calculated using ODE78 and plots the differences.
    """
    # Extract satellite parameters from TLE data
    satellite_name = tle_data[0][0]
    A, m, C_D = get_satellite_params(satellite_name)
    if A is None or m is None:
        print(f"Satellite parameters for {satellite_name} are missing.")
        return

    # Compute initial position and velocity
    jd, fr = jday(year, month, day, hour, minute, second)
    satellite = Satrec.twoline2rv(tle_data[0][1], tle_data[0][2])
    e, r, v = satellite.sgp4(jd, fr)
    if e != 0:
        print(f"Error in SGP4 propagation for {satellite_name}: {e}")
        return

    y0 = np.concatenate((r, v))  # Initial state vector: position + velocity

    for algorithm_name, csv_file in csv_files.items():
        df = pd.read_csv(csv_file)

        if 'Time (s)' not in df.columns or not {' X (km)', ' Y (km)', ' Z (km)'}.issubset(df.columns):
            print(f"CSV file '{csv_file}' is missing required columns.")
            continue

        csv_time = df['Time (s)'].values
        csv_positions = df[[' X (km)', ' Y (km)', ' Z (km)']].values

        time_points = []
        position_differences = []

        for i, t in enumerate(csv_time):
            # Compute position using ODE78
            results = ODE78.ode78(
                lambda t, y: satellite_motion(t, y ),
                [0, t],
                y0,
                atol=1e-16,
                rtol=1e-19
            )

            # Extract the position from ODE78 results at time `t`
            current_position = results[-1][1:4]

            # Get the position from the CSV for comparison
            csv_position = csv_positions[i]

            # Calculate the position difference
            position_difference = np.linalg.norm(np.array(current_position) - np.array(csv_position))

            # Store the time and position difference
            time_points.append(t)
            position_differences.append(position_difference)

        # Plot the differences
        ax.plot(time_points, position_differences, label=f"{algorithm_name} vs. ODE78", linestyle='--')

def plot_execution_time_from_csv(file_path):
    """
    Plots execution times for algorithms from the provided CSV file.
    """
    df = pd.read_csv(file_path)
    if 'Algorithm' not in df.columns or 'Execution Time (s)' not in df.columns:
        print("CSV file does not have the required columns: 'Algorithm' and 'Execution Time (s)'")
        return

    algorithms = df['Algorithm'].values
    execution_times = df['Execution Time (s)'].values

    plt.figure(figsize=(10, 6))
    plt.bar(algorithms, execution_times, color='skyblue')

    plt.title("Execution Time of Algorithms", fontsize=14)
    plt.xlabel("Algorithms", fontsize=12)
    plt.ylabel("Execution Time (s)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Sample TLE data
tle_data = [
    ("ASBM-2",
     "1 60423U 24143B   24286.24012071  .00000053  00000-0  00000-0 0  9996",
     "2 60423  62.3237  68.6665 5330361 267.8786  33.3421  1.50421044  2382"),
]
year, month, day, hour, minute, second = 2024, 10, 12, 12, 0, 0

csv_files = {
    "RK4": "Satellite_RK4_final_positions.csv",
    "RK8": "Satellite_RK8_final_positions.csv",
    "ODE45": "Satellite_ODE45_final_positions.csv",
    "ODE78": "Satellite_ODE78_final_positions.csv",
    "ODE113": "Satellite_ODE113_final_positions.csv"
}

# Plotting position differences using ODE78
fig, ax = plt.subplots(figsize=(10, 6))
compare_csv_with_ode78_and_plot(ax, tle_data, year, month, day, hour, minute, second, csv_files)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Time (s)')
ax.set_xlim(10e-6, None)
ax.set_ylim(10e-16, None)
ax.set_ylabel('Position Difference (km)')
ax.set_title(f'Position Difference vs. Time (CSV vs. ODE78)')
ax.grid(True)
ax.legend()
plt.show()
# Plotting execution times
plot_execution_time_from_csv('algorithm_execution_times.csv')