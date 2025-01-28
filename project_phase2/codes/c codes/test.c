#include "adaptive_picard_chebyshev.h"
#include "c_functions.h"
#include "EGM2008.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

FILE* fID;

int main() {


    // Initialize variables for reading CSV
    char buffer[256];  // Smaller buffer size
    int num_rows = 0;

    // Arrays for storing data
    double times[89791];  // Adjust size as needed
    double initial_positions[89791][3];
    double initial_velocities[89791][3];
    double final_positions[89791][3];

    // Open the CSV file for reading
    FILE* csv_file = fopen("sgp4_results.csv", "r");
    if (csv_file == NULL) {
        printf("Error opening CSV file\n");
        return 1;
    }

    // Skip the first line (header)
    fgets(buffer, 256, csv_file);

    // Read the CSV file row by row with a smaller buffer size
    while (fgets(buffer, 256, csv_file)) {
        // Parse the line into respective fields
        sscanf(buffer, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
               &times[num_rows],
               &initial_positions[num_rows][0], &initial_positions[num_rows][1], &initial_positions[num_rows][2],
               &initial_velocities[num_rows][0], &initial_velocities[num_rows][1], &initial_velocities[num_rows][2],
               &final_positions[num_rows][0], &final_positions[num_rows][1], &final_positions[num_rows][2]);

        // Print the parsed data (matrix form) after reading each row
        printf("Row %d - Time: %lf\n", num_rows, times[num_rows]);
        printf("Initial Position: [%lf, %lf, %lf]\n", initial_positions[num_rows][0], initial_positions[num_rows][1], initial_positions[num_rows][2]);
        printf("Initial Velocity: [%lf, %lf, %lf]\n", initial_velocities[num_rows][0], initial_velocities[num_rows][1], initial_velocities[num_rows][2]);
        printf("Final Position: [%lf, %lf, %lf]\n", final_positions[num_rows][0], final_positions[num_rows][1], final_positions[num_rows][2]);
        printf("-----------------------------\n");

        num_rows++;
    }

    fclose(csv_file);

    printf("Finished reading data, now starting APC calculations.\n");

    // APC variables
    double t0 = 0.0;        // Initial Time (s)
    double dt;              // Time step based on CSV file data
    double deg = 70.0;      // Gravity Degree (max 100)
    double tol = 1.0e-15;    // Tolerance

    // Initialize Output Variables
    double* Soln;
    int soln_size;
    double Feval[2] = { 0.0 };

    // declare the initial position
    double y0[3];
    memcpy(y0, initial_positions[1], 3 * sizeof(double));
    printf("y0=%12lf\n", y0[0]);

    //declare the initial velocity
    double v0[3];
    memcpy(v0, initial_velocities[1], 3 * sizeof(double));
    printf("v0=%12lf\n", v0[0]);

    // Open the CSV file for writing the APC results
    FILE* output_csv = fopen("apc_results.csv", "w");
    if (output_csv == NULL) {
        printf("Error opening output CSV file\n");
        return 1;
    }

    // Open the CSV file for writing final positions
    FILE* final_positions_csv = fopen("final_positions_results.csv", "w");
    if (final_positions_csv == NULL) {
        printf("Error opening final positions CSV file\n");
        return 1;
    }

    // Write the header for the output CSV files
    fprintf(output_csv, "Time (s),Error (km)\n");
    fprintf(final_positions_csv, "Time (s),X (km),Y (km),Z (km)\n");
    fflush(output_csv);  // Ensure header is written immediately
    fflush(final_positions_csv);  // Ensure header is written immediately

    // Loop through the data and run APC for each time interval
    for (int i = 0; i < num_rows; i++) {

        // Set time step for this run (final time from the CSV)
        double tf = times[i];
        dt = tf - t0;

        if (tf < 0.1)
            tf = 0.0001;
        if (dt < 0.1)
            dt = 0.0001;
        // Estimate solution size based on time interval
        soln_size = (int)(1.1 * (tf / dt));
        if (soln_size == 1) {
            soln_size = 2;
        }

        // Allocate memory for the solution
        Soln = calloc(soln_size * 6, sizeof(double));  // Position (km) & Velocity (km/s)

        printf("t0=%lf\n", t0);
        printf("dt=%lf\n", dt);
        printf("tf=%lf\n", tf);

        // Call APC for the current satellite data
        adaptive_picard_chebyshev(initial_positions[i], initial_velocities[i], t0, tf, dt, deg, tol, soln_size, Feval, Soln);
        printf("Solution Array (Soln):\n");
        for (int i = 0; i < soln_size; i++) {
            printf("Soln[%d]: [%12.12lf, %12.12lf, %12.12lf, %12.12lf, %12.12lf, %12.12lf]\n",
                   i,
                   Soln[i * 6 + 0],  // X Position
                   Soln[i * 6 + 1],  // Y Position
                   Soln[i * 6 + 2],  // Z Position
                   Soln[i * 6 + 3],  // X Velocity
                   Soln[i * 6 + 4],  // Y Velocity
                   Soln[i * 6 + 5]); // Z Velocity
        }

        // Extract the final position from the last row (corSrect indices for position)
        double apc_final_position[3] = {
                Soln[ID2(soln_size, 1, soln_size)],  // X position at final time
                Soln[ID2(soln_size, 2, soln_size)],  // Y position at final time
                Soln[ID2(soln_size, 3, soln_size)]   // Z position at final time
        };

        // Compare APC final position with SGP4 final position from CSV
        double sgp4_final_position[3] = {
                final_positions[i][0],
                final_positions[i][1],
                final_positions[i][2]
        };

        double error = sqrt(
                pow(apc_final_position[0] - sgp4_final_position[0], 2) +
                pow(apc_final_position[1] - sgp4_final_position[1], 2) +
                pow(apc_final_position[2] - sgp4_final_position[2], 2)
        );

        // Print position difference (error) before writing to the file
        printf("Position Difference (Error) at Time %lf s: %lf km\n", tf, error);

        // Write the time and error to the output CSV file
        fprintf(output_csv, "%12lf,%12lf\n", tf, error);
        fflush(output_csv);  // Ensure each result is written immediately

        // Write the final position to the final_positions_results.csv file
        // Write the final position to the final_positions_results.csv file
        fprintf(final_positions_csv, "%.12lf,%.12lf,%.12lf,%.12lf\n",
                tf, apc_final_position[0], apc_final_position[1], apc_final_position[2]);
        fflush(final_positions_csv);  // Ensure the final position is written immediately

        // Free memory for the solution
        free(Soln);
    }

    // Close the CSV files
    fclose(output_csv);
    fclose(final_positions_csv);

    return 0;
}