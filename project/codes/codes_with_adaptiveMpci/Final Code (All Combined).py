import datetime
import math
import numpy as np
from sgp4.api import Satrec, jday
import RK4
import RK8
import ODE45
import ODE78
import ODE113
import MPCI
import time
import matplotlib.pyplot as plt

# Constants
EARTH_RADIUS_KM = 6378.137  # WGS84 Earth radius at the equator in kilometers
comparison_results = []
def plot_average_results(averages):
    algorithms = list(averages.keys())
    avg_position_differences = [averages[algo]['avg_position_difference'] for algo in algorithms]
    avg_execution_times = [averages[algo]['avg_execution_time'] for algo in algorithms]

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Bar chart for average position differences
    ax1.bar(algorithms, avg_position_differences, color='skyblue', label='Average Position Difference (km)')
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Average Position Difference (km)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Line plot for average execution times
    ax2 = ax1.twinx()
    ax2.plot(algorithms, avg_execution_times, color='red', marker='o', linestyle='-', label='Average Execution Time (s)')
    ax2.set_ylabel('Average Execution Time (s)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Add a title and a grid
    plt.title(f"Average Algorithm Comparison Across All Satellites")
    ax1.grid(True)

    # Show the plot
    fig.tight_layout()
    plt.show()
def plot_comparison_bar_chart_with_styled_table(comparison_results, params_dict):
    satellites = list(set(result['object_name'] for result in comparison_results))
    
    # Define the column headers for the parameters
    col_labels = ["Step Size", "Tolerance", "Max Step Size", "Min Step Size", "Nodes"]

    for satellite in satellites:
        satellite_results = [result for result in comparison_results if result['object_name'] == satellite]
        
        algorithms = [result['algorithm'] for result in satellite_results]
        position_differences = [result['position_difference'] for result in satellite_results]
        execution_times = [result['execution_time'] for result in satellite_results]

        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Bar chart for position differences
        ax1.bar(algorithms, position_differences, color='skyblue', label='Position Difference (km)')
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Position Difference (km)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Line plot for execution times
        ax2 = ax1.twinx()
        ax2.plot(algorithms, execution_times, color='red', marker='o', linestyle='-', label='Execution Time (s)')
        ax2.set_ylabel('Execution Time (s)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Add a title and a grid
        plt.title(f"Algorithm Comparison for {satellite}")
        ax1.grid(True)

        # Create a table for parameters
        cell_text = []
        
        for algo in algorithms:
            if algo in params_dict:
                row_data = params_dict[algo]
                cell_text.append(row_data)
        
        # Add table below the plot with enhanced styling
        table = ax1.table(cellText=cell_text, rowLabels=algorithms, colLabels=col_labels, cellLoc='center', loc='bottom', bbox=[0.0, -0.5, 1.0, 0.4])
        
        # Styling the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # Style the headers
        for (i, j), cell in table.get_celld().items():
            cell.set_text_props(weight='bold')
            if i == 0:  # Header row
                cell.set_facecolor('#40466e')
                cell.set_text_props(color='white')
            cell.set_edgecolor('gray')
            cell.set_linewidth(1)
        
        # Adjust layout to accommodate the table
        plt.subplots_adjust(left=0.2, bottom=0.4)

        # Show the plot
        fig.tight_layout()
        plt.show()

# Example parameter dictionary (this should match the actual parameters used)
params_dict = {
    'RK4': ["0.1", "-", "-", "-", "-"],
    'RK8': ["0.1", "-", "-", "-", "-"],
    'ODE45': ["-", "1e-6", "0.1", "1e-6", "-"],
    'ODE78': ["0.1", "-", "-", "-", "-"],
    'ODE113': ["-", "1e-6", "0.1", "1e-6", "-"],
    'MPCI': ["0.1", "1e-6", "-", "-", "30"]
}

def compute_averages(comparison_results):
    # Initialize dictionaries to hold sums and counts
    averages = {}
    algorithms = set(result['algorithm'] for result in comparison_results)

    for algo in algorithms:
        algo_results = [result for result in comparison_results if result['algorithm'] == algo]
        avg_position_diff = np.mean([result['position_difference'] for result in algo_results])
        avg_execution_time = np.mean([result['execution_time'] for result in algo_results])
        averages[algo] = {
            'avg_position_difference': avg_position_diff,
            'avg_execution_time': avg_execution_time
        }
    
    return averages
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
            distance_from_surface = math.sqrt(r[0]**2 + r[1]**2 + r[2]**2) - EARTH_RADIUS_KM
            lat, lon, alt = eci_to_geodetic(r[0], r[1], r[2])
            results.append({
                'object_name': object_name,
                'tle_line1': tle_line1,
                'tle_line2': tle_line2,
                'position': {'x': r[0], 'y': r[1], 'z': r[2]},
                'velocity': {'vx': v[0], 'vy': v[1], 'vz': v[2]},
                'distance_from_surface': distance_from_surface,
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
def write_results_to_file(filename, algorithm_name, results, position_difference, velocity_difference, elapsed_time):
    """
    Writes the results of the last 2 seconds and comparison with SGP4 to a text file.
    """
    with open(filename, "w") as file:
        file.write(f'********************* {algorithm_name} Algorithm ******************************\n')
        file.write(f"Results for the last 2 seconds (Time | Position X, Y, Z | Velocity X, Y, Z):\n")
        file.write(f"{'Time':^8} {'PosX':^12} {'PosY':^12} {'PosZ':^12} {'VelX':^12} {'VelY':^12} {'VelZ':^12}\n")
        file.write("-" * 80 + "\n")

        for row in results:
            if row[0] >= 29.0:  # Only write the results for the last two seconds (t >= 58.0)
                file.write(f"{row[0]:^8.2f} {row[1]:^12.2f} {row[2]:^12.2f} {row[3]:^12.2f} {row[4]:^12.2f} {row[5]:^12.2f} {row[6]:^12.2f}\n")
        
        # Write position and velocity differences
        file.write(f"\nPosition difference between {algorithm_name} and SGP4 after 30 seconds: {position_difference:.6f} km\n")
        file.write(f"Velocity difference between {algorithm_name} and SGP4 after 30 seconds: {velocity_difference:.6f} km/s\n")
        file.write(f"{algorithm_name} execution time: {elapsed_time:.6f} seconds\n")
    
    print(f'Results saved to {filename}\n')



def compare_algorithms(r0, v0, algorithm_name, algorithm_module, func_name, tle_data, year, month, day, hour, minute, second, **kwargs):
    print(f'********************* {algorithm_name} Algorithm ******************************')
    start_time = time.time()

    y0 = np.concatenate((r0, v0))  # Initial state vector
    tspan = (0, 30)  # Time range from 0 to 60 seconds
    mu = 398600  # Standard gravitational parameter for Earth

    # Run the algorithm
    results = getattr(algorithm_module, func_name)(lambda t, y: algorithm_module.satellite_motion(t, y, mu), tspan, y0, **kwargs)
    elapsed_time = time.time() - start_time

    # Print detailed results for the last two seconds (Time | Position X, Y, Z | Velocity X, Y, Z)
    print("Results for the last 2 seconds (Time | Position X, Y, Z | Velocity X, Y, Z):")
    print(f"{'Time':^8} {'PosX':^12} {'PosY':^12} {'PosZ':^12} {'VelX':^12} {'VelY':^12} {'VelZ':^12}")
    print("-" * 80)
    
    LastResult = None
    for row in results:
        if row[0] >= 29.0:  # Only print the results for the last two seconds (t >= 58.0)
            print(f"{row[0]:^8.2f} {row[1]:^12.2f} {row[2]:^12.2f} {row[3]:^12.2f} {row[4]:^12.2f} {row[5]:^12.2f} {row[6]:^12.2f}")
            LastResult = np.array([row[0], row[1], row[2], row[3], row[4], row[5], row[6]])

    # If results are available for the last time step (t=60s), calculate position difference from SGP4
    if LastResult is not None:
        final_position = LastResult[1:4]
        final_velocity = LastResult[4:7]

        # Compute the new results after 1 hour for comparison
        new_results = compute_position_velocity_and_geodetic(tle_data, year, month, day, hour, minute , second+30.0)
        for new_result in new_results:
            r0_new_sgp4 = np.array([new_result['position']['x'], new_result['position']['y'], new_result['position']['z']])
            v0_new_sgp4 = np.array([new_result['velocity']['vx'], new_result['velocity']['vy'], new_result['velocity']['vz']])

        position_difference = abs(np.linalg.norm(final_position) - np.linalg.norm(r0_new_sgp4))
        velocity_difference = abs(np.linalg.norm(final_velocity) - np.linalg.norm(v0_new_sgp4))

        # Store results in comparison_results for further processing
        comparison_results.append({
            'object_name': tle_data[0][0],
            'algorithm': algorithm_name,
            'position_difference': position_difference,
            'velocity_difference': velocity_difference,
            'execution_time': elapsed_time
        })
        # Write the results to a file
        filename = f"{algorithm_name}_results.txt"
        write_results_to_file(filename, algorithm_name, results, position_difference, velocity_difference, elapsed_time)

        print(f"\nPosition difference between {algorithm_name} and SGP4 after 30 seconds: {position_difference:.6f} km")
        print(f"Velocity difference between {algorithm_name} and SGP4 after 30 seconds: {velocity_difference:.6f} km/s")
        print(f"{algorithm_name} execution time: {elapsed_time:.6f} seconds\n")
    else:
        print('No result found for time > 29.0.')




# Sample TLE data
tle_data = [

    
    ("0 VANGUARD 2",
     "1 00011U 59001A   24246.83325614  .00001550  00000-0  80246-3 0  9991",
     "2 00011  32.8774  31.2424 1454929 308.7215  39.1836 11.88679839457759"),

    ("0 VANGUARD 3",
     "1    20U 59007A   24246.93286316  .00005525  00000-0  22227-2 0  9993",
     "2    20  33.3419  91.4400 1648564 162.5066 203.9956 11.59594611408823"),

    ("0 EXPLORER 7",
     "1    22U 59009A   24247.15714936  .00018726  00000-0  10176-2 0  9993",
     "2    22  50.2776  39.5867 0107531  52.8645 308.2073 15.09621260456132"),

    ("0 TIROS 1",
     "1 00029U 60002B   24246.68644685  .00003123  00000-0  46893-3 0  9996",
     "2 00029  48.3764 241.8644 0023510 339.7284  20.2693 14.76716864445613"),

    ("0 TRANSIT 2A",
     "1 00045U 60007A   24246.84648773  .00001461  00000-0  37678-3 0  9993",
     "2 00045  66.6928 284.5414 0259413 275.0747  82.0794 14.34883511286411"),

    ("0 VELA 1",
     "1 00692U 63039C   24243.97226542 -.00000558  00000-0  00000-0 0  9999",
     "2 00692  19.5921  37.9251 5936118 213.2807 359.3793  0.22154371 11724")
    
]

current_time_utc = datetime.datetime.utcnow()
year, month, day, hour, minute, second = (current_time_utc.year, current_time_utc.month, current_time_utc.day, 
                                          current_time_utc.hour, current_time_utc.minute, current_time_utc.second)

for sat_data in tle_data:
    # Compute initial position and velocity for the satellite
    results = compute_position_velocity_and_geodetic([sat_data], year, month, day, hour, minute, second)
    result = results[0]
    if 'error' not in result:
        r0 = np.array([result['position']['x'], result['position']['y'], result['position']['z']])
        v0 = np.array([result['velocity']['vx'], result['velocity']['vy'], result['velocity']['vz']])
        
        # Compare all algorithms for this satellite
        step_size = 0.1
        tol = 1e-6
        h_min = 1e-6
        compare_algorithms(r0, v0, "RK4", RK4, "rk4", [sat_data], year, month, day, hour, minute, second, h=step_size)
        compare_algorithms(r0, v0, "RK8", RK8, "RK8", [sat_data], year, month, day, hour, minute, second, h=step_size)
        compare_algorithms(r0, v0, "ODE45", ODE45, "ode45", [sat_data], year, month, day, hour, minute, second, tol=tol, hmax=step_size, hmin=h_min)
        compare_algorithms(r0, v0, "ODE78", ODE78, "ode78", [sat_data], year, month, day, hour, minute, second, h=step_size)
        compare_algorithms(r0, v0, "ODE113", ODE113, "adams_bashforth_moulton_adaptive", [sat_data], year, month, day, hour, minute, second, tol=tol, hmax=step_size, hmin=h_min)
        compare_algorithms(r0, v0, "MPCI", MPCI, "MPCI", [sat_data], year, month, day, hour, minute, second, h_init=step_size, tol=tol ,max_iter=100,N_init=30)
    else:
        print(f"Error computing initial position and velocity for satellite {sat_data[0]}: {result['error']}")

# Print comparison results
print("\nPosition and Velocity Differences (compared to SGP4) \n")
print(f"{'Satellite':<20}{'Algorithm':<15}{'Position Difference (km)':<30}{'Velocity Difference (km/s)':<30}{'Execution Time (s)':<30}")
print("-" * 120)
for result in comparison_results:
    print(f"{result['object_name']:<20}{result['algorithm']:<15}{result['position_difference']:<30.6f}{result['velocity_difference']:<30.6f}{result['execution_time']:<30.6f}")


# Call the function to plot the comparison bar charts with a professionally styled table
plot_comparison_bar_chart_with_styled_table(comparison_results, params_dict)
# Calculate and plot averages
averages = compute_averages(comparison_results)
plot_average_results(averages)


