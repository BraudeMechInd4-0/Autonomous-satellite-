import os
import pandas as pd
import matplotlib.pyplot as plt

# Folder paths
base_folder = "csvfiles"
apci_file = os.path.join(base_folder, "apci_algorithm_execution_times.csv")

# List of satellites
satellites = ["ASBM-2", "STARLINK-1341", "IRIDIUM 33 DEB", "QIANFAN-4", "SKYNET 4C"]

# Read APCI execution times (common for all satellites)
apci_data = pd.read_csv(apci_file)

# Ensure APCI data only includes the specified satellites
apci_filtered = apci_data[apci_data["satelite_name"].isin(satellites)]

# Color mapping for algorithms
color_map = {
    "RK4": "blue",
    "RK8": "orange",
    "ODE45": "green",
    "ODE78": "red",
    "ODE113": "purple"
}

# Function to read algorithm execution times for a satellite (8 segments, 16 segments)
def read_algorithm_execution_times(satellite_name, filename):
    file_path = os.path.join(base_folder, satellite_name, filename + ".csv")
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        # Extract only the first word of the algorithm name (e.g., "RK4" from "RK4_with_segments")
        data["Algorithm"] = data["Algorithm"].str.split("_").str[0]
        return data
    else:
        print(f"File not found: {file_path}")
        return pd.DataFrame()

# Plot algorithm execution times for each satellite and file type
for satellite in satellites:
    for file_type in ["algorithm_execution_times_8_16", "algorithm_execution_times_16_16"]:
        satellite_data = read_algorithm_execution_times(satellite, file_type)
        if not satellite_data.empty:
            title_suffix = "8 segments, 16 points" if "8_16" in file_type else "16 segments, 16 points"
            plt.figure(figsize=(8, 5))
            
            # Assign colors based on the algorithm name
            colors = satellite_data["Algorithm"].map(color_map)
            
            plt.bar(satellite_data["Algorithm"], satellite_data["Execution Time (s)"], color=colors)
            plt.title(f"{satellite} Execution Times ({title_suffix})", fontsize=14)
            plt.xlabel("Algorithms", fontsize=12)
            plt.ylabel("Execution Time (s)", fontsize=12)
            plt.xticks(rotation=45, fontsize=10)
            plt.tight_layout()
            plt.show()

# Plot APCI execution times
plt.figure(figsize=(10, 6))
plt.bar(apci_filtered["satelite_name"], apci_filtered["Execution Time (hours)"], color="skyblue")
plt.xlabel("Satellites", fontsize=12)
plt.ylabel("Execution Time (hours)", fontsize=12)
plt.title("APCI Execution Times Across Satellites", fontsize=14)
plt.xticks(rotation=45, fontsize=10)
plt.tight_layout()
plt.show()
