import math
import numpy as np
from sgp4.api import Satrec, jday
import ODE78Baseline
import matplotlib.pyplot as plt
import pandas as pd  # Import pandas to read the CSV file

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

def gauss_lobatto_points(n, a, b):
    points = np.array([-np.cos(np.pi * i / (n - 1)) for i in range(n)], dtype=np.float64)
    scaled_points = 0.5 * (b - a) * (np.array(points) + 1) + a
    return scaled_points
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

def write_to_excel(data, filename):
    # Create a DataFrame for better readability
    df = pd.DataFrame(
        data,
        columns=[
            "Time",
            "Position_X", "Position_Y", "Position_Z",
            "Velocity_VX", "Velocity_VY", "Velocity_VZ"
        ]
    )
    
    # Save to Excel
    df.to_excel(filename, index=False, float_format="%.18f")

def basealgorithm(r0, v0, tle_data, year, month, day, hour, minute, second, A, m, C_D, apc_csv_file):

    y0 = np.concatenate((r0, v0))  # Initial state vector: position + velocity
    tspan=[0,1000]
    # Create a figure for plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    baseResults = np.array(
        ODE78Baseline.ode78(
            lambda t, y: satellite_motion_only_gravity(t, y, mu=398600),
            tspan,
            y0,
            atol=1e-16,
            rtol=1e-12,
            h=0.001,
            h_max=0.01
        ),
        dtype=np.float64
    )

    write_to_excel(baseResults, "baseResults.xlsx")



def satellite_motion_only_gravity(t, y, mu=398600):
    """
    Simplified satellite motion considering only gravity.
    """
    r = np.array(y[:3])  # Position vector
    v = np.array(y[3:])  # Velocity vector
    r_norm = np.linalg.norm(r)  # Magnitude of the position vector
    a = -mu / r_norm**3 * r  # Acceleration vector
    dydt = np.concatenate((v, a))  # Combine velocity and acceleration
    return dydt



# Sample TLE data
tle_data = [

("SKYNET 4C",
 "1 20776U 90079A   24273.86593681  .00000122  00000-0  00000-0 0  9996",
 "2 20776  13.5422 354.0676 0003521 217.5807 142.6885  1.00279641124611")
]
# Sample date and initial position, velocity
date_tuple = (2024, 10, 12, 12, 0, 0)  # Fixed date and time
year, month, day, hour, minute, second = date_tuple

# Compute position and velocity based on TLE data
results = compute_position_velocity_and_geodetic([tle_data[0]], year, month, day, hour, minute, second)
result = results[0]
print("result=",result)

if 'error' not in result:
    r0 = np.array([result['position']['x'], result['position']['y'], result['position']['z']], dtype=np.float64)
    v0 = np.array([result['velocity']['vx'], result['velocity']['vy'], result['velocity']['vz']], dtype=np.float64)

    
    A, m, C_D = get_satellite_params(tle_data[0][0])
    
    if A is not None and m is not None:
        # Plot all algorithms and the CSV vs. SGP4 comparison together
        basealgorithm(r0, v0, tle_data, year, month, day, hour, minute, second, A, m, C_D, '')