import numpy as np
import pandas as pd

def apci(file_name):
    # Read the CSV file using pandas
    data = pd.read_csv(file_name)
    
    # Extract the 'Time (s)' column as tout
    tout = data['Time (s)'].to_numpy()
    
    # Extract the 'X (km)', 'Y (km)', and 'Z (km)' columns as yout
    yout = data[['X (km)', 'Y (km)', 'Z (km)']].to_numpy()
    
    # Return the stacked result
    return np.column_stack((tout, yout))

'''
# Example usage
file_name = "apci_final_positions_results.csv"
result = apci(file_name)
print(result[0])
'''