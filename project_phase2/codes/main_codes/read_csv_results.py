import numpy as np
import pandas as pd

def read_results(file_name):
    """
    Reads a CSV file containing results and returns the required data as a NumPy array.

    Parameters:
    file_name - The name of the CSV file to read.

    Returns:
    results - A NumPy array with [Time (s), X (km), Y (km), Z (km)].
    """
    try:
        # Read the CSV file using pandas
        data = pd.read_csv(file_name)

        # Clean column names to remove invisible characters or trailing spaces
        data.columns = data.columns.str.strip()

        # Validate required columns
        required_columns = ['Time (s)', 'X (km)', 'Y (km)', 'Z (km)']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing column '{col}' in file {file_name}. Available columns: {data.columns.tolist()}")

        # Extract the 'Time (s)' column as tout
        tout = data['Time (s)'].to_numpy(dtype=float)

        # Extract the 'X (km)', 'Y (km)', and 'Z (km)' columns as yout
        yout = data[['X (km)', 'Y (km)', 'Z (km)']].to_numpy(dtype=float)

        # Return the stacked result
        return np.column_stack((tout, yout))

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_name}")
    except ValueError as e:
        raise ValueError(f"Error reading file {file_name}: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error while reading file {file_name}: {e}")
