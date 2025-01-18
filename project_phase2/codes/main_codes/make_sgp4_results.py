import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
import RK4
import RK8
import ODE45
import ODE78
import ODE113
from sgp4.api import Satrec, jday
from numpy.polynomial.legendre import leggauss
import time
import csv  # Import CSV for reading APC results

# Constants
EARTH_RADIUS_KM = 6378.137  # WGS84 Earth radius at the equator in kilometers
MU_EARTH = 398600  # Gravitational parameter for Earth in km^3/s^2

# Function to convert ECI to geodetic coordinates
def eci_to_geodetic(x, y, z):
    r = math.sqrt(x**2 + y**2 + z**2)
    lon = math.degrees(math.atan2(y, x))
    lat = math.degrees(math.atan2(z, math.sqrt(x**2 + y**2)))
    alt = r - EARTH_RADIUS_KM
    return lat, lon, alt

# Function to compute position and velocity using SGP4
def compute_position_velocity_and_geodetic(tle_data, year, month, day, hour, minute, second):
    jd, fr = jday(year, month, day, hour, minute, second)
    results = []
    for object_name, tle_line1, tle_line2 in tle_data:
        satellite = Satrec.twoline2rv(tle_line1, tle_line2)
        e, r, v = satellite.sgp4(jd, fr)
        if e == 0:
            results.append({
                'object_name': object_name,
                'position': np.array([r[0], r[1], r[2]]),
                'velocity': np.array([v[0], v[1], v[2]])
            })
        else:
            results.append({'object_name': object_name, 'error': e})
    return results

# Function to write results to file with precise time calculations
def write_sgp4_results_to_file(filename, r0, v0, time_points, tle_data_list, date_tuple):
    with open(filename, 'w') as f:
        f.write("Time (s), Initial Position (x, y, z), Initial Velocity (vx, vy, vz), Final Position (x, y, z)\n")
        
        # For each time point, calculate the SGP4 result
        for t in time_points:
            # Calculate time with precision
            precise_time = round(t, 12)
            
            # Get initial position and velocity from TLE data
            initial_position = r0
            initial_velocity = v0

            # Compute final position after precise_time seconds using SGP4
            year, month, day, hour, minute, second = date_tuple
            # Add precise_time to the base datetime
            precise_second = second + precise_time
            jd, fr = jday(year, month, day, hour, minute, precise_second)
            satellite = Satrec.twoline2rv(*tle_data_list[0][1:])
            e, r, v = satellite.sgp4(jd, fr)

            if e != 0:  # Skip in case of SGP4 errors
                continue

            final_position = np.array(r)

            # Write to file
            f.write(f"{precise_time:.12f}, {initial_position[0]:.12f}, {initial_position[1]:.12f}, {initial_position[2]:.12f}, "
                    f"{initial_velocity[0]:.12f}, {initial_velocity[1]:.12f}, {initial_velocity[2]:.12f}, "
                    f"{final_position[0]:.12f}, {final_position[1]:.12f}, {final_position[2]:.12f}\n")


# Function to propagate using various algorithms
def propagate_with_algorithm(algorithm_name, algorithm_module, func_name, r0, v0, tspan, **kwargs):
    y0 = np.concatenate((r0, v0))  # Initial state vector
    mu = MU_EARTH
    results = getattr(algorithm_module, func_name)(lambda t, y: satellite_motion_only_gravity(t, y, mu), tspan, y0, **kwargs)
    return results

# Define the satellite motion function (gravity only)
def satellite_motion_only_gravity(t, y, mu=MU_EARTH):
    r = np.array(y[:3])  # Position vector
    v = np.array(y[3:])  # Velocity vector
    r_norm = np.linalg.norm(r)  # Magnitude of the position vector
    a = -mu / r_norm**3 * r  # Gravitational acceleration
    dydt = np.concatenate((v, a))  # Combine velocity and acceleration
    return dydt

# Function to read APC results from CSV file
def read_apc_results(apc_filename):
    apc_times = []
    apc_errors = []
    with open(apc_filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            apc_times.append(float(row[0]))  # Time in seconds
            apc_errors.append(float(row[1]))  # Error in km
    return apc_times, apc_errors

# Plot error vs time on log-log scale
def plot_error_comparison(results_dict, time_values, apc_times=None, apc_errors=None):
    plt.figure(figsize=(10, 8))
    
    # Plot other algorithms
    for algorithm_name, error_values in results_dict.items():
        plt.plot(time_values, error_values, label=algorithm_name)
    
    # Add APC data to the plot as a line, not points
    if apc_times and apc_errors:
        plt.plot(apc_times, apc_errors, label='APC', linestyle='--', color='black')  # Removed 'marker' argument

    # Set scales and labels
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Error (km)', fontsize=14)
    plt.title('Error Comparison Over Time', fontsize=16)
    plt.grid(True, which="both", ls="--")
    plt.legend(loc='upper left')
    plt.show()

def gauss_lobatto_points(n, a, b):
    points = [-np.cos(np.pi * i / (n - 1)) for i in range(n)]
    scaled_points = 0.5 * (b - a) * (np.array(points) + 1) + a
    return scaled_points

def generate_time_points(total_time, num_segments, points_per_segment):
    time_points = []
    segment_length = total_time / num_segments
    for i in range(num_segments):
        t_start = i * segment_length
        t_end = t_start + segment_length
        points = gauss_lobatto_points(points_per_segment, t_start, t_end)
        time_points.extend(points)
    return np.array(time_points)

# Compare algorithms and collect error data
def compare_algorithms(r0, v0, tle_data_list, date_tuple, time_points, algorithms):
    time_values = []
    results_dict = {}

    # Get the reference (SGP4) result at each time point
    for t in time_points:
        year, month, day, hour, minute, second = date_tuple  # Extract the reference time
        reference_result = compute_position_velocity_and_geodetic(tle_data_list, year, month, day, hour, minute, second + int(t))[0]
        if 'error' in reference_result:
            continue  # Skip if there's an error in SGP4 calculation
        reference_position = reference_result['position']
        
        # For each algorithm, calculate error at this time
        for algorithm_name, algorithm_module, func_name, kwargs in algorithms:
            results = propagate_with_algorithm(algorithm_name, algorithm_module, func_name, r0, v0, (0, t), **kwargs)
            final_position = results[-1][:3]  # Final position from the algorithm's result
            error = np.linalg.norm(final_position - reference_position)  # Compute error
            if algorithm_name not in results_dict:
                results_dict[algorithm_name] = []
            results_dict[algorithm_name].append(error)
        
        time_values.append(t)

    # Read APC results from file
    apc_times, apc_errors = read_apc_results('apc_results.csv')

    # Plot the results with APC included
    plot_error_comparison(results_dict, time_values, apc_times, apc_errors)

# Generate time points
total_time = 1_000_000  # 1 million seconds
num_segments = 16
points_per_segment = 32
time_points = generate_time_points(total_time, num_segments, points_per_segment)

# Example TLE data as a list of tuples
tle_data_list = [
    ("ASBM-2",
     "1 60423U 24143B   24286.24012071  .00000053  00000-0  00000-0 0  9996",
     "2 60423  62.3237  68.6665 5330361 267.8786  33.3421  1.50421044  2382"),
]
date_tuple = (2024, 10, 12, 12, 0, 0)

# Call the function correctly
results = compute_position_velocity_and_geodetic(tle_data_list, *date_tuple)

# Now you can use results to extract the position and velocity for further calculations
r0 = results[0]['position']
v0 = results[0]['velocity']

# Write SGP4 results to file
output_filename = "sgp4_results.csv"
write_sgp4_results_to_file(output_filename, r0, v0, time_points, tle_data_list, date_tuple)

# List of algorithms to compare
algorithms = [
    ("RK4", RK4, "rk4", {"h": 0.1}),
    ("RK8", RK8, "RK8", {"h": 0.1}),
    ("ODE45", ODE45, "ode45", {"tol": 1e-6, "hmax": 0.1, "hmin": 1e-6}),  # Added hmax and hmin
    ("ODE78", ODE78, "ode78", {"h": 0.1}),
    ("ODE113", ODE113, "ODE113", {"tol": 1e-6}),
]
