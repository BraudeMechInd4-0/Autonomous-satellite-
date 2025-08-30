#!/usr/bin/env python3
"""
Script to generate LaTeX execution time comparison tables from satellite propagation results.
Uses known satellite names from JSON and known configurations to avoid parsing issues.
"""

import pandas as pd
import json
import os
from pathlib import Path

def load_satellites_from_json(json_file="../satellites.json"):
    """Load satellite names from JSON file."""
    try:
        with open(json_file, 'r') as f:
            satellites = json.load(f)
        return [sat['name'] for sat in satellites]
    except FileNotFoundError:
        print(f"Error: {json_file} not found!")
        return []

def load_execution_times_from_csv(filename):
    """Load execution times from a single CSV file and return as dictionary."""
    try:
        df = pd.read_csv(filename)
        times_dict = {}
        
        for _, row in df.iterrows():
            algorithm = row['Algorithm']
            # Use the total time column (last column should be total)
            if 'Total Time (s)' in df.columns:
                total_time = row['Total Time (s)']
            else:
                # If no total column, use the last numeric column
                numeric_cols = [col for col in df.columns if col != 'Algorithm']
                if numeric_cols:
                    total_time = row[numeric_cols[-1]]  # Assume last column is total
                else:
                    continue
            
            times_dict[algorithm] = total_time
            
        return times_dict
    
    except Exception as e:
        print(f"Warning: Could not load {filename}: {e}")
        return {}

def create_latex_table(satellite_name, N, data_dict):
    """
    Create a LaTeX table for a specific satellite and N (number of points).
    
    Args:
        satellite_name: Name of the satellite
        N: Number of points used (called N in the paper)
        data_dict: Dictionary with structure {delta: {algorithm: time}} where delta is number of segments
    """
    
    algorithms = ['RK4', 'RK8', 'ODE45', 'ODE78', 'ODE113', 'MPCI']
    delta_list = [8, 16, 32]  # Known values (number of segments)
    
    # Start LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\begin{tabular}{|c|c|c|c|c|c|c|}")
    latex_lines.append("\\hline")
    latex_lines.append("$\\delta$ & RK4 & RK8 & ODE45 & ODE78 & ODE113 & MPCI \\\\")
    latex_lines.append("\\hline")
    
    # Add data rows
    for delta in delta_list:
        row = [str(delta)]
        
        if delta in data_dict:
            for algorithm in algorithms:
                if algorithm in data_dict[delta]:
                    time_val = data_dict[delta][algorithm]
                    row.append(f"{time_val:.3f}")
                else:
                    row.append("N/A")
        else:
            row.extend(["N/A"] * len(algorithms))
        
        latex_lines.append(" & ".join(row) + " \\\\")
        latex_lines.append("\\hline")
    
    latex_lines.append("\\end{tabular}")
    
    # Fix the satellite name for LaTeX outside the f-string
    latex_satellite_name = satellite_name.replace('_', '\\_')
    latex_lines.append(f"\\caption{{Execution Times (seconds) for {latex_satellite_name} with $N={N}$}}")
    
    # Fix the satellite name for label outside the f-string  
    label_name = satellite_name.lower().replace(' ', '_').replace('-', '_')
    latex_lines.append(f"\\label{{tab:{label_name}_N{N}}}")
    
    latex_lines.append("\\end{table}")
    latex_lines.append("")
    
    return "\n".join(latex_lines)

def generate_latex_tables():
    """Generate LaTeX execution time tables for all satellites and configurations."""
    
    # Load satellite names from JSON
    satellite_names = load_satellites_from_json()
    
    if not satellite_names:
        print("No satellite names loaded from JSON!")
        return
    
    print(f"Loaded {len(satellite_names)} satellites: {satellite_names}")
    
    # Known configurations
    num_points_list = [8, 16, 32]
    num_segments_list = [8, 16, 32]
    
    all_latex_content = []
    all_latex_content.append("% Satellite Propagation Algorithm Execution Time Tables")
    all_latex_content.append("% Generated automatically")
    all_latex_content.append("")
    
    # Process each satellite
    for satellite_name in satellite_names:
        print(f"\nProcessing satellite: {satellite_name}")
        
        # Process each number of points
        for num_points in num_points_list:
            print(f"  Processing {num_points} points...")
            
            # Collect data for all segment configurations
            data_dict = {}
            
            for num_segments in num_segments_list:
                # Construct expected filename
                # Based on your main file: SATELLITENAME_NUMSEGMENTS_NUMPOINTS_execution_times.csv
                filename = f"{satellite_name}_{num_segments}_{num_points}_execution_times.csv"
                
                if os.path.exists(filename):
                    print(f"    Found: {filename}")
                    times_dict = load_execution_times_from_csv(filename)
                    if times_dict:
                        data_dict[num_segments] = times_dict
                else:
                    print(f"    Missing: {filename}")
            
            # Generate LaTeX table if we have any data
            if data_dict:
                latex_table = create_latex_table(satellite_name, num_points, data_dict)
                all_latex_content.append(latex_table)
                print(f"    Generated table for {satellite_name} - {num_points} points")
            else:
                print(f"    No data found for {satellite_name} with {num_points} points")
    
    # Save single LaTeX file with all tables
    output_file = "all_execution_time_tables.tex"
    with open(output_file, 'w') as f:
        f.write("\n".join(all_latex_content))
    
    print(f"\nAll LaTeX tables saved to: {output_file}")
    print(f"Total tables generated: {len([line for line in all_latex_content if 'begin{table}' in line])}")



def generate_summary_statistics():
    """Generate summary statistics across all configurations."""
    
    satellite_names = load_satellites_from_json()
    if not satellite_names:
        return
    
    N_list = [8, 16, 32]  # Number of points (called N in paper)
    delta_list = [8, 16, 32]  # Number of segments (called delta in paper)
    algorithms = ['RK4', 'RK8', 'ODE45', 'ODE78', 'ODE113', 'MPCI']
    
    # Collect all data
    all_data = []
    
    for satellite_name in satellite_names:
        for N in N_list:
            for delta in delta_list:
                filename = f"{satellite_name}_{delta}_{N}_execution_times.csv"
                
                if os.path.exists(filename):
                    times_dict = load_execution_times_from_csv(filename)
                    
                    for algorithm in algorithms:
                        if algorithm in times_dict:
                            all_data.append({
                                'Satellite': satellite_name,
                                'N': N,
                                'Delta': delta,
                                'Algorithm': algorithm,
                                'Time': times_dict[algorithm]
                            })
    
    if not all_data:
        print("No data collected for summary statistics")
        return
    
    df = pd.DataFrame(all_data)
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Average time by algorithm
    print("\nAverage Execution Time by Algorithm:")
    avg_times = df.groupby('Algorithm')['Time'].agg(['mean', 'std', 'min', 'max'])
    print(avg_times.round(4))
    
    # Best algorithm by configuration
    print("\nFastest Algorithm by Configuration:")
    fastest = df.loc[df.groupby(['N', 'Delta'])['Time'].idxmin()]
    print(fastest[['N', 'Delta', 'Algorithm', 'Time']].round(4).to_string(index=False))

if __name__ == "__main__":
    print("Generating LaTeX execution time comparison tables...")
    print("Using known satellite names from satellites.json")
    print("Using known configurations: δ (segments)=[8,16,32], N (points)=[8,16,32]")
    print()
    
    generate_latex_tables()
    generate_summary_statistics()
    print("\nDone!")