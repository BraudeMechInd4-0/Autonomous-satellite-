import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sgp4.api import Satrec, jday

def compare_csv_with_sgp4_and_plot(ax, tle_data, year, month, day, hour, minute, second, csv_files, apci_file):
    """
    Compares multiple algorithms' position data from CSV files with SGP4 and plots the position differences.
    Includes the APCI algorithm results.
    """
    jd, fr = jday(year, month, day, hour, minute, second)
    satellite = Satrec.twoline2rv(tle_data[0][1], tle_data[0][2])
    
    # Plot results for other algorithms
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
            e, r_sgp4, _ = satellite.sgp4(jd, fr + t / 86400.0)
            if e == 0:
                csv_position = csv_positions[i]
                position_difference = np.linalg.norm(np.array(csv_position) - np.array(r_sgp4))
                time_points.append(t)
                position_differences.append(position_difference)
            else:
                time_points.append(t)
                position_differences.append(float('nan'))

        ax.plot(time_points, position_differences, label=f"{algorithm_name} vs. SGP4", linestyle='--')

    # Plot APCI data
    apci_data = pd.read_csv(apci_file)
    if 'Time (s)' in apci_data.columns and {'X (km)', 'Y (km)', 'Z (km)'}.issubset(apci_data.columns):
        apci_time = apci_data['Time (s)'].values
        apci_positions = apci_data[['X (km)', 'Y (km)', 'Z (km)']].values

        apci_differences = []
        for i, t in enumerate(apci_time):
            e, r_sgp4, _ = satellite.sgp4(jd, fr + t / 86400.0)
            if e == 0:
                apci_position = apci_positions[i]
                apci_difference = np.linalg.norm(np.array(apci_position) - np.array(r_sgp4))
                apci_differences.append(apci_difference)
            else:
                apci_differences.append(float('nan'))

        ax.plot(apci_time, apci_differences, label="APCI vs. SGP4", linestyle='--', color='red' )
    else:
        print(f"APCI file '{apci_file}' is missing required columns.")

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

# APCI data file path
apci_file = "final_positions_results_low_system.csv"

# Plotting position differences
fig, ax = plt.subplots(figsize=(10, 6))
compare_csv_with_sgp4_and_plot(ax, tle_data, year, month, day, hour, minute, second, csv_files, apci_file)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Time (s)')
ax.set_xlim(10e-2, None)
ax.set_ylim(10e-16, None)
ax.set_ylabel('Position Difference (km)')
ax.set_title(f'Position Difference vs. Time for Multiple Algorithms Including APCI')
ax.grid(True)
ax.legend()
plt.show()

# Plotting execution times
plot_execution_time_from_csv('algorithm_execution_times.csv')
