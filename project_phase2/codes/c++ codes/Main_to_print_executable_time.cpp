#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <vector>
#include <tuple>
#include <iomanip>
#include <algorithm>
#include <string>
#include <sstream>
#include <ctime>
#include <filesystem>
#include "CommonFunctions.h"
#include "RK4.h"
#include "RK8.h"
#include "ODE45.h"
#include "ODE78.h"
#include "ODE113.h"
#include "coefficients78.h"
#include <numeric>

// Define constants
const double EARTH_RADIUS_KM = 6378.137;  // Earth's radius in kilometers
const int NUM_SEGMENTS = 8;
const int NUM_GAUSS_LOBATTO_POINTS = 16;  // Number of points per segment, can be adjusted


void save_results_to_csv(const std::string& satellite_name, const std::string& algorithm_name,
                         const std::vector<double>& time_points, const std::vector<double>& position_differences) {
    std::string directory = "results";
    std::filesystem::create_directory(directory);

    std::string filename = directory + "/" + satellite_name + "_" + algorithm_name + "_results.csv";
    std::ofstream file(filename);

    if (file.is_open()) {
        file << "Time (s), Error (km)\n";
        for (size_t i = 0; i < time_points.size(); ++i) {
            file << std::fixed << std::setprecision(18)
                 << time_points[i] << "," << position_differences[i] << "\n";
        }
        std::cout << "Results saved to " << filename << std::endl;
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }

    file.close();
}

// Function to save final positions to a CSV file
void save_final_positions_to_csv(const std::string& satellite_name, const std::string& algorithm_name,
                                 const std::vector<double>& time_points,
                                 const std::vector<std::vector<double>>& positions) {
    std::string directory = "results";
    std::filesystem::create_directory(directory);

    std::string filename = directory + "/" + satellite_name + "_" + algorithm_name +
                       "_final_positions_" + std::to_string(NUM_SEGMENTS) +
                       "_" + std::to_string(NUM_GAUSS_LOBATTO_POINTS) + ".csv";
    std::ofstream file(filename);

    if (file.is_open()) {
        file << "Time (s), X (km), Y (km), Z (km)\n";
        for (size_t i = 0; i < time_points.size(); ++i) {
            file << std::fixed << std::setprecision(18)
                 << time_points[i] << "," << positions[i][0] << ","
                 << positions[i][1] << "," << positions[i][2] << "\n";
        }
        std::cout << "Final positions saved to " << filename << std::endl;
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }

    file.close();
}


// Function to write results to a CSV file
void writeToCSV(const std::vector<double>& tout, const std::vector<std::vector<double>>& yout, const std::string& filename, bool write_header = false) {
    std::ofstream file;
    if (write_header) {
        file.open(filename);  // Overwrite mode (write header)
    } else {
        file.open(filename, std::ios::app);  // Append mode
    }

    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }

    // Set precision to 18 decimal places
    file << std::fixed << std::setprecision(18);

    // Write header (only if write_header is true)
    if (write_header) {
        file << "Time (s),X (km),Y (km),Z (km)" << std::endl;
    }

    // Write data
    for (size_t i = 0; i < tout.size(); ++i) {
        file << tout[i] << "," << yout[i][0] << "," << yout[i][1] << "," << yout[i][2] << std::endl;
    }

    file.close();
}


// Function definitions for each algorithm (rk4, rk8, ode45, ode78, ODE113)
// Assume these functions are implemented in other files as per your setup
extern std::vector<std::vector<double>>  rk4(
std::vector<double>(*odefun)(double, const std::vector<double>&, double, double, double), // Updated signature
const std::vector<double>& t_gauss_lobatto,
const std::vector<double>& y0,
double A,  // Cross-sectional area
double m,  // Satellite mass
double C_D
);
extern std::vector<std::vector<double>> RK8(
    std::vector<double>(*f)(double, const std::vector<double>&, double , double , double , double),
    const std::vector<double>& t_gauss_lobatto, const std::vector<double>& Y0, double h,
    double A,  // Cross-sectional area
    double m,  // Satellite mass
    double C_D // Drag coefficient
);

extern std::vector<std::vector<double>> ode78(
    std::function<std::vector<double>(double, const std::vector<double>&)> ode_func,
    const std::vector<double>& t_span,
    const std::vector<double>& y0,
    const std::vector<double>& b,  // 8th-order weights
    const std::vector<double>& bh, // 7th-order weights
    const std::vector<double>& c,  // Time nodes (Gauss-Lobatto points)
    const std::map<int, std::map<int, double>>& a,  // Coupling coefficients (Butcher tableau)
    double rtol ,
    double atol
);

void compare_RK4_algorithm_with_segments(
    const std::string& algorithm_name,
    std::vector<std::vector<double>> (*algorithm_func)(
        std::vector<double>(*)(double, const std::vector<double>&, double, double, double),
        const std::vector<double>&,
        const std::vector<double>&,
        double, double, double
    ),
    const std::vector<double>& r0,
    const std::vector<double>& v0,
    double total_time,
    double orbital_period,
    double A,
    double m,
    double C_D
) {
    // ✅ Combine initial position and velocity into y0
    std::vector<double> y0 = r0;
    y0.insert(y0.end(), v0.begin(), v0.end());  // Append velocity to position

    // Simulation parameters
    double total_time1 = 0.0;
    int num_segments = NUM_SEGMENTS;
    int n_points = NUM_GAUSS_LOBATTO_POINTS;

    double total_execution_time = 0.0;
    std::vector<double> execution_times;
    std::vector<std::vector<double>> final_positions;
    std::vector<double> time_points;

    // Loop until total simulation time exceeds 1,000,000 seconds
    while (total_time1 < 1000000) {
        total_time1 += orbital_period;
        std::cout << "Current total simulation time = " << total_time1 << std::endl;

        double segment_time = orbital_period / num_segments;

        for (int i = 0; i < num_segments; ++i) {
            double t_start = total_time1 - orbital_period + i * segment_time;
            double t_end = t_start + segment_time;


            // Generate Gauss-Lobatto points for the current segment
            std::vector<double> gauss_lobatto_tspan = gaussLobattoPoints(n_points, t_start, t_end);

            // 🚀 Propagate using RK4
            auto start_time = std::chrono::high_resolution_clock::now();
            auto results = rk4(a_c_func, gauss_lobatto_tspan, y0, A, m, C_D);
            auto end_time = std::chrono::high_resolution_clock::now();

            // Measure execution time
            std::chrono::duration<double> elapsed = end_time - start_time;
            total_execution_time += elapsed.count();

            // ✅ Update y0 with the last state from the current segment
            y0 = results.back();

            // 📝 Store results for CSV saving
            for (size_t j = 0; j < gauss_lobatto_tspan.size(); ++j) {
                time_points.push_back(gauss_lobatto_tspan[j]);
                final_positions.push_back({results[j][0], results[j][1], results[j][2]});
            }

            // Store the execution time for the segment
            execution_times.push_back(elapsed.count());
        }
    }

    // ✅ Save execution times and results to CSV
    save_final_positions_to_csv("Satellite", algorithm_name, time_points, final_positions);

    // ✅ Print summary
    std::cout << algorithm_name << " total execution time: " << total_execution_time << " seconds\n";
    std::cout << "Final position after " << total_time << " seconds: "
              << y0[0] << ", " << y0[1] << ", " << y0[2] << "\n";
}






void compare_RK8_algorithm_with_segments(
    const std::string& algorithm_name,
    std::vector<std::vector<double>> (*algorithm_func)(
        std::vector<double>(*)(double, const std::vector<double>&, double, double, double, double),
        const std::vector<double>&,
        const std::vector<double>&,
        double,
        double,
        double,
        double
    ),
    const std::vector<double>& r0,
    const std::vector<double>& v0,
    double total_time,
    double orbital_period,
    double h = 0.1,   // Step size
    double A = 12.0,  // Cross-sectional area
    double m = 2000,   // Satellite mass
    double C_D = 2.2  // Drag coefficient
) {
    // ✅ Combine initial position and velocity into y0
    std::vector<double> y0 = r0;
    y0.insert(y0.end(), v0.begin(), v0.end());  // Append velocity to position

    // Simulation parameters
    double total_time1 = 0.0;
    int num_segments = NUM_SEGMENTS;
    int n_points = NUM_GAUSS_LOBATTO_POINTS;

    double total_execution_time = 0.0;
    std::vector<double> execution_times;
    std::vector<std::vector<double>> final_positions;
    std::vector<double> time_points;

    // Loop until total simulation time exceeds 1,000,000 seconds
    while (total_time1 < 1000000) {
        total_time1 += orbital_period;
        std::cout << "Current total simulation time = " << total_time1 << std::endl;

        double segment_time = orbital_period / num_segments;

        for (int i = 0; i < num_segments; ++i) {
            double t_start = total_time1 - orbital_period + i * segment_time;
            double t_end = t_start + segment_time;
            std::cout << "Segment " << i + 1 << ": t_start = " << t_start << ", t_end = " << t_end << std::endl;

            // Generate Gauss-Lobatto points for the current segment
            std::vector<double> gauss_lobatto_tspan = gaussLobattoPoints(n_points, t_start, t_end);

            // 🚀 Propagate using RK8
            auto start_time = std::chrono::high_resolution_clock::now();
            auto results = RK8(a_c_func_new, gauss_lobatto_tspan, y0, h, A, m, C_D);
            auto end_time = std::chrono::high_resolution_clock::now();

            // Measure execution time
            std::chrono::duration<double> elapsed = end_time - start_time;
            total_execution_time += elapsed.count();

            // ✅ Update y0 with the last state from the current segment
            y0 = results.back();

            // 📝 Store results for CSV saving
            for (size_t j = 0; j < gauss_lobatto_tspan.size(); ++j) {
                time_points.push_back(gauss_lobatto_tspan[j]);
                final_positions.push_back({results[j][0], results[j][1], results[j][2]});
            }

            // Store the execution time for the segment
            execution_times.push_back(elapsed.count());
        }
    }

    // ✅ Save execution times and results to CSV
    save_final_positions_to_csv("Satellite", algorithm_name, time_points, final_positions);

    // ✅ Print summary
    std::cout << algorithm_name << " total execution time: " << total_execution_time << " seconds\n";
    std::cout << "Final position after " << total_time << " seconds: "
              << y0[0] << ", " << y0[1] << ", " << y0[2] << "\n";
}




void compare_ODE45_algorithm_with_segments(
    const std::string& algorithm_name,
    std::vector<std::vector<double>> (*algorithm_func)(
        std::vector<double>(*)(double, const std::vector<double>&, double, double, double),  // ODE function pointer (a_c_func)
        const std::vector<double>&,  // Time span (Gauss-Lobatto points)
        const std::vector<double>&,  // Initial state vector
        double,  // Cross-sectional area (A)
        double,  // Mass (m)
        double,  // Drag coefficient (C_D)
        double,  // Relative tolerance (rtol)
        double   // Absolute tolerance (atol)
    ),
    const std::vector<double>& r0,  // Initial position vector
    const std::vector<double>& v0,  // Initial velocity vector
    double total_time,              // Total simulation time
    double orbital_period,          // Orbital period
    double A,                       // Cross-sectional area
    double m,                       // Mass
    double C_D,                     // Drag coefficient
    double rtol,                    // Relative tolerance
    double atol                     // Absolute tolerance
)
{

    // Calculate the number of orbits and segment duration
    int num_orbits = static_cast<int>(total_time / orbital_period); // Number of orbits
    int num_segments = NUM_SEGMENTS; // Number of segments per orbit
    double segment_duration = orbital_period / num_segments; // Duration of each segment

    std::vector<double> y0 = r0; // Initial position
    y0.insert(y0.end(), v0.begin(), v0.end()); // Append initial velocity to form state vector

    double total_execution_time = 0.0;
    int total_points = 0; // Variable to count total Gauss-Lobatto points used

    // Vectors to store execution times, final positions, and time points
    std::vector<double> execution_times;
    std::vector<std::vector<double>> final_positions;
    std::vector<double> time_points;

    // Outer loop for orbits
    for (int orbit = 0; orbit < num_orbits+1; ++orbit) {
        for (int segment = 0; segment < num_segments; ++segment) {
            double t_start = orbit * orbital_period + segment * segment_duration;
            double t_end = t_start + segment_duration;

            // Generate Gauss-Lobatto points for the current segment
            std::vector<double> segment_time_points = gaussLobattoPoints(NUM_GAUSS_LOBATTO_POINTS, t_start, t_end);

            // Increment total_points by the number of Gauss-Lobatto points
            total_points += segment_time_points.size();

            auto start_time = std::chrono::high_resolution_clock::now();

            // Execute the ODE45 algorithm for all Gauss-Lobatto points in this segment
            std::vector<std::vector<double>> results = algorithm_func(
            a_c_func,               // ✅ Use aerodynamic model with drag and J2 effects
            segment_time_points,    // Gauss-Lobatto time points
            y0,                     // Initial state vector
            A,                      // Cross-sectional area
            m,                      // Mass
            C_D,                    // Drag coefficient
            rtol,                   // Relative tolerance
            atol                    // Absolute tolerance
        );

            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end_time - start_time;

            // Accumulate total execution time for the segment
            double execution_time = elapsed.count();
            total_execution_time += execution_time;

            // Iterate through each result and store positions and times
            for (size_t i = 0; i < segment_time_points.size(); ++i) {
                time_points.push_back(segment_time_points[i]); // Store time point
                final_positions.push_back({results[i][0], results[i][1], results[i][2]}); // Store position
            }

            // Update y0 to the last state of this segment
            y0 = results.back();

            // Store execution time for the segment
            execution_times.push_back(execution_time);
        }
    }


    // Save final positions to a CSV file (time point-level granularity)
    save_final_positions_to_csv("Satellite", algorithm_name, time_points, final_positions);

    // Print total execution time
    std::cout << algorithm_name << " total execution time: " << total_execution_time << " seconds\n";

    // Print the final position after the last segment
    std::cout << "Final position after " << total_time << " seconds: "
              << y0[0] << ", " << y0[1] << ", " << y0[2] << "\n";

    // Print the total number of Gauss-Lobatto points used
    std::cout << "Total number of Gauss-Lobatto points used: " << total_points << "\n";
}



void compare_ODE78_algorithm_with_segments(
    const std::string& algorithm_name,
    std::vector<std::vector<double>> (*algorithm_func)(
        std::function<std::vector<double>(double, const std::vector<double>&, double, double, double)>, // ODE function pointer
        const std::vector<double>&,  // Time span (Gauss-Lobatto points)
        const std::vector<double>&,  // Initial state vector
        const std::vector<double>&,  // 8th-order weights (b)
        const std::vector<double>&,  // 7th-order weights (bh)
        const std::vector<double>&,  // Time nodes (c)
        const std::map<int, std::map<int, double>>&,  // Butcher tableau (a)
        double,  // Relative tolerance (rtol)
        double,  // Absolute tolerance (atol)
        double,  // Cross-sectional area (A)
        double,  // Mass (m)
        double   // Drag coefficient (C_D)
    ),
    const std::vector<double>& r0,  // Initial position vector
    const std::vector<double>& v0,  // Initial velocity vector
    double total_time,              // Total simulation time
    double orbital_period,          // Orbital period
    double A,                       // Cross-sectional area
    double m,                       // Mass
    double C_D,                     // Drag coefficient
    double rtol,                    // Relative tolerance
    double atol                     // Absolute tolerance
) {
    int num_orbits = static_cast<int>(total_time / orbital_period);  // Number of orbits
    int num_segments = NUM_SEGMENTS;  // Number of segments per orbit
    double segment_duration = orbital_period / num_segments;  // Segment duration

    std::vector<double> y0 = r0;
    y0.insert(y0.end(), v0.begin(), v0.end());  // Combine position and velocity

    double total_execution_time = 0.0;
    int total_points = 0;

    std::vector<double> execution_times;
    std::vector<std::vector<double>> final_positions;
    std::vector<double> time_points;

    auto func = [](double t, const std::vector<double>& y, double A, double m, double C_D) {
        return a_c_func(t, y, A, m, C_D);  // Aerodynamic model with drag
    };

    for (int orbit = 0; orbit < num_orbits+1; ++orbit) {
        for (int segment = 0; segment < num_segments; ++segment) {
            double t_start = orbit * orbital_period + segment * segment_duration;
            double t_end = t_start + segment_duration;

            std::vector<double> segment_time_points = gaussLobattoPoints(NUM_GAUSS_LOBATTO_POINTS, t_start, t_end);
            total_points += segment_time_points.size();
            // Convert c from map<int, double> to vector<double>
            std::vector<double> c_vector(c.size());
            for (const auto& pair : c) {
                c_vector[pair.first - 1] = pair.second;  // Convert map to vector, adjusting index
            }
            auto start_time = std::chrono::high_resolution_clock::now();

            // ✅ Updated call to `ode78` with new parameters
            std::vector<std::vector<double>> results = algorithm_func(
                func,                   // ODE function
                segment_time_points,    // Gauss-Lobatto points
                y0,                     // Initial state
                b,                      // 8th-order weights
                bh,                     // 7th-order weights
                c_vector,                      // Time nodes
                a,                      // Butcher tableau
                rtol,                   // Relative tolerance
                atol,                   // Absolute tolerance
                A,                      // Cross-sectional area
                m,                      // Mass
                C_D                     // Drag coefficient
            );

            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end_time - start_time;

            total_execution_time += elapsed.count();

            // Process results for each Gauss-Lobatto point
            for (const auto& result : results) {
                double t_point = result[0];
                std::vector<double> current_position(result.begin() + 1, result.begin() + 4);  // Extract position

                time_points.push_back(t_point);            // Store time point
                final_positions.push_back(current_position); // Store final position
            }

            // Update y0 for the next segment with the last state of this segment
            y0[0] = results.back()[1];
            y0[1] = results.back()[2];
            y0[2] = results.back()[3];
            y0[3] = results.back()[4];
            y0[4] = results.back()[5];
            y0[5] = results.back()[6];
            execution_times.push_back(elapsed.count());
        }
    }

    // Save execution times and results
    save_final_positions_to_csv("Satellite", algorithm_name, time_points, final_positions);

    std::cout << algorithm_name << " total execution time: " << total_execution_time << " seconds\n";
    std::cout << "Final position after " << total_time << " seconds: "
              << y0[0] << ", " << y0[1] << ", " << y0[2] << "\n";
    std::cout << "Total number of Gauss-Lobatto points used: " << total_points << "\n";
    std::cout << "Total number of Gauss-Lobatto points used: " << total_points << "\n";
}





void compare_ODE113_algorithm_with_segments(
        const std::string& algorithm_name,
        const std::vector<double>& r0,
        const std::vector<double>& v0,
        double tol,
        double hmax,
        double hmin,
        double mu,
        double orbital_period,
        double total_time,
        double A,
        double m,
        double C_D) {
    int num_orbits = static_cast<int>(total_time / orbital_period);
    int num_segments = NUM_SEGMENTS;
    double segment_duration = orbital_period / num_segments;

    std::vector<double> y0 = r0;
    y0.insert(y0.end(), v0.begin(), v0.end());

    double total_execution_time = 0.0;
    int total_points = 0;

    std::vector<double> execution_times;
    std::vector<std::vector<double>> final_positions;
    std::vector<double> time_points;

    for (int orbit = 0; orbit < num_orbits+1; ++orbit) {
        for (int segment = 0; segment < num_segments; ++segment) {
            double t_start = orbit * orbital_period + segment * segment_duration;
            double t_end = t_start + segment_duration;

            // Generate Gauss-Lobatto points
            std::vector<double> segment_time_points = gaussLobattoPoints(NUM_GAUSS_LOBATTO_POINTS, t_start, t_end);
            total_points += segment_time_points.size();

            auto segment_start_time = std::chrono::high_resolution_clock::now();

            // Execute ODE113 for the current segment
            ODE113Result results = ODE113(
                a_c_func_new,
                segment_time_points,
                y0,
                mu,
                tol,
                tol, // Absolute tolerance
                hmax,
                hmin,
                A,
                m,
                C_D
            );

            auto segment_end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> segment_elapsed = segment_end_time - segment_start_time;

            // Store execution time
            execution_times.push_back(segment_elapsed.count());
            total_execution_time += segment_elapsed.count();

            // Store results for each Gauss-Lobatto point
            for (size_t i = 0; i < results.time.size(); ++i) {
                time_points.push_back(results.time[i]);
                final_positions.push_back({results.values[i][0], results.values[i][1], results.values[i][2]});
            }

            // Update y0 for the next segment
            y0 = results.values.back();
        }
    }

    // Save results to CSV
    save_final_positions_to_csv("Satellite", algorithm_name, time_points, final_positions);

    // Print statistics
    std::cout << algorithm_name << " total execution time: " << total_execution_time << " seconds\n";
    std::cout << "Final position after " << total_time << " seconds: "
              << y0[0] << ", " << y0[1] << ", " << y0[2] << "\n";
    std::cout << "Total number of points used: " << total_points << "\n";
}





int main() {
    // Initial conditions
    std::vector<double> r0 = { -23760.532622384497, -33715.879108559609, -8669.038042343847};  // Initial position vector
    std::vector<double> v0 = {2.539009331841, -1.701743118962, -0.342990147826}; // Initial velocity vector

    // Define parameters for ODE113
    double tol = 1e-16;   // Tolerance
    double hmax = 0.01;   // Maximum step size
    double hmin = 1e-10;  // Minimum step size
    double mu = 398600.4418;  // Gravitational parameter for Earth
    double total_time = 1000000;  // Total simulation time in seconds
    // Compute the orbital period
    double orbital_period =       86162.08174373899;

    // Satellite parameters (Area, Mass, Drag Coefficient)
    double A = 10.0;
    double m = 1250;
    double C_D = 2.2;

    // File to save execution times
    std::ofstream exec_time_file("algorithm_execution_times.csv");
    if (!exec_time_file.is_open()) {
        std::cerr << "Error: Could not open file to save execution times.\n";
        return 1;
    }
    exec_time_file << "Algorithm,Execution Time (s)\n";

    // Timer for RK4 with segments
    auto start = std::chrono::high_resolution_clock::now();
    compare_RK4_algorithm_with_segments(
    "RK4",
    rk4,          // Pass the updated rk4 function
    r0, v0,       // Initial position and velocity
    total_time,
    orbital_period,
    A, m, C_D     // Pass the satellite parameters
);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    exec_time_file << "RK4_with_segments," << elapsed.count() << "\n";
    std::cout << "RK4_with_segments execution time: " << elapsed.count() << " seconds\n";

    // Timer for RK8 with segments
    start = std::chrono::high_resolution_clock::now();
    double step_size = 0.1;  // Step size for RK8
    // ✅ Corrected RK8 call with all necessary parameters
    compare_RK8_algorithm_with_segments(
        "RK8",  // Algorithm name
        RK8,                  // Pass the RK8 function pointer
        r0,                   // Initial position vector
        v0,                   // Initial velocity vector
        total_time,           // Total simulation time
        orbital_period,       // Orbital period
        step_size,            // Integration step size (h)
        A,                    // Cross-sectional area
        m,                    // Satellite mass
        C_D                   // Drag coefficient
    );
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    exec_time_file << "RK8_with_segments," << elapsed.count() << "\n";
    std::cout << "RK8_with_segments execution time: " << elapsed.count() << " seconds\n";

    // Timer for ODE45 with segments
    start = std::chrono::high_resolution_clock::now();
    const double MU = 398600;    // Gravitational parameter
    const double RTOL = 1e-16;    // Relative tolerance
    const double ATOL = 1e-16;    // Absolute tolerance
    // Correct call to compare ODE45 algorithm with segments
    compare_ODE45_algorithm_with_segments(
        "ODE45",  // Algorithm name for logging
        ode45,                  // ODE45 function
        r0,                     // Initial position
        v0,                     // Initial velocity
        total_time,             // Total simulation time
        orbital_period,         // Orbital period
        A,                      // Cross-sectional area
        m,                      // Mass
        C_D,                    // Drag coefficient
        RTOL,                   // Relative tolerance
        ATOL                    // Absolute tolerance
    );
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    exec_time_file << "ODE45_with_segments," << elapsed.count() << "\n";
    std::cout << "ODE45_with_segments execution time: " << elapsed.count() << " seconds\n";

    // Timer for ODE78 with segments
    start = std::chrono::high_resolution_clock::now();
    double rtol = 1e-3;
    double atol = 1e-6;
    // ✅ Correct call to compare_ODE78_algorithm_with_segments
    compare_ODE78_algorithm_with_segments(
        "ODE78",  // Algorithm name for result logging
        ode78,                  // Pass the updated ode78 function
        r0,                     // Initial position vector
        v0,                     // Initial velocity vector
        total_time,             // Total simulation time
        orbital_period,         // Orbital period
        A,                      // Cross-sectional area
        m,                      // Mass
        C_D,                    // Drag coefficient
        rtol,                   // Relative tolerance
        atol                    // Absolute tolerance
    );
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    exec_time_file << "ODE78_with_segments," << elapsed.count() << "\n";
    std::cout << "ODE78_with_segments execution time: " << elapsed.count() << " seconds\n";

    // Timer for ODE113 with segments
    start = std::chrono::high_resolution_clock::now();
    compare_ODE113_algorithm_with_segments(
    "ODE113", r0, v0, 1e-6, 100, 1e-3, mu, orbital_period, 970000 ,A,m,C_D
);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    exec_time_file << "ODE113_with_segments," << elapsed.count() << "\n";
    std::cout << "ODE113_with_segments execution time: " << elapsed.count() << " seconds\n";

    // Close the execution time file
    exec_time_file.close();

    return 0;
}
