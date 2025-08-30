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
#include "MPCI.h"
#include "coefficients78.h"
#include <numeric>

// Define constants
const double EARTH_RADIUS_KM = 6378.137; // Earth's radius in kilometers
// const int NUM_SEGMENTS = 8;
// const int NUM_GAUSS_LOBATTO_POINTS = 16;  // Number of points per segment, can be adjusted

void save_results_to_csv(const std::string &satellite_name, const std::string &algorithm_name,
                         const std::vector<double> &time_points, const std::vector<double> &position_differences)
{
    std::string directory = "results";
    std::filesystem::create_directory(directory);

    std::string filename = directory + "/" + satellite_name + "_" + algorithm_name + "_results.csv";
    std::ofstream file(filename);

    if (file.is_open())
    {
        file << "Time (s), Error (km)\n";
        for (size_t i = 0; i < time_points.size(); ++i)
        {
            file << std::fixed << std::setprecision(18)
                 << time_points[i] << "," << position_differences[i] << "\n";
        }
        std::cout << "Results saved to " << filename << std::endl;
    }
    else
    {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }

    file.close();
}

// Function to save final positions to a CSV file
void save_final_positions_to_csv(const std::string &satellite_name, const std::string &algorithm_name,
                                 const std::vector<double> &time_points,
                                 const std::vector<std::vector<double>> &positions, int num_segments, int n_points)
{
    std::string directory = "results";
    std::filesystem::create_directory(directory);

    std::string filename = directory + "/" + satellite_name + "_" + algorithm_name +
                           "_final_positions_" + std::to_string(num_segments) +
                           "_" + std::to_string(n_points) + ".csv";
    std::ofstream file(filename);

    if (file.is_open())
    {
        file << "Time (s), X (km), Y (km), Z (km)\n";
        for (size_t i = 0; i < time_points.size(); ++i)
        {
            file << std::fixed << std::setprecision(18)
                 << time_points[i] << "," << positions[i][0] << ","
                 << positions[i][1] << "," << positions[i][2] << "\n";
        }
        std::cout << "Final positions saved to " << filename << std::endl;
    }
    else
    {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }

    file.close();
}

// Function to write results to a CSV file
void writeToCSV(const std::vector<double> &tout, const std::vector<std::vector<double>> &yout, const std::string &filename, bool write_header = false)
{
    std::ofstream file;
    if (write_header)
    {
        file.open(filename); // Overwrite mode (write header)
    }
    else
    {
        file.open(filename, std::ios::app); // Append mode
    }

    if (!file.is_open())
    {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }

    // Set precision to 18 decimal places
    file << std::fixed << std::setprecision(18);

    // Write header (only if write_header is true)
    if (write_header)
    {
        file << "Time (s),X (km),Y (km),Z (km)" << std::endl;
    }

    // Write data
    for (size_t i = 0; i < tout.size(); ++i)
    {
        file << tout[i] << "," << yout[i][0] << "," << yout[i][1] << "," << yout[i][2] << std::endl;
    }

    file.close();
}

// Structure to hold parsed command line arguments
struct SimulationParams
{
    std::string satellite_name;
    std::vector<double> r0;
    std::vector<double> v0;
    double A;
    double m;
    double C_D;
    int num_segments;
    int num_gauss_lobatto_points;
    double total_time;
    double absTol;
    double relTol;
};

bool parseCommandLineArgs(int argc, char *argv[], SimulationParams &params)
{
    // Check if we have the correct number of arguments
    if (argc != 16)
    {
        std::cerr << "Usage: " << argv[0] << "Usage: " << argv[0] << "<name> <r0_x> <r0_y> <r0_z> <v0_x> <v0_y> <v0_z> <A> <m> <C_D> <num_segments> <num_gauss_lobatto_points> <tmax>" << std::endl;
        std::cerr << "Example: " << argv[0] << "Example: " << argv[0] << "  " << std::endl;
        return false;
    }

    try
    {
        // Initialize vectors
        params.r0.resize(3);
        params.v0.resize(3);

        // Parse name
        params.satellite_name = std::string(argv[1]);

        // Parse r0 vector (position)
        params.r0[0] = std::stod(argv[2]); // r0_x
        params.r0[1] = std::stod(argv[3]); // r0_y
        params.r0[2] = std::stod(argv[4]); // r0_z

        // Parse v0 vector (velocity)
        params.v0[0] = std::stod(argv[5]); // v0_x
        params.v0[1] = std::stod(argv[6]); // v0_y
        params.v0[2] = std::stod(argv[7]); // v0_z

        // Parse satellite parameters
        params.A = std::stod(argv[8]);    // Cross-sectional area
        params.m = std::stod(argv[9]);    // Mass
        params.C_D = std::stod(argv[10]); // Drag coefficient

        params.num_segments = std::stoi(argv[11]);
        params.num_gauss_lobatto_points = std::stoi(argv[12]);
        params.total_time = std::stoi(argv[13]); // Fixed total time for simulation

        params.absTol = std::stoi(argv[14]);
        params.relTol = std::stoi(argv[15]);

        // Print parsed parameters for verification
        std::cout << "=== Simulation Parameters ===" << std::endl;
        std::cout << "Initial Position (r0): [" << params.r0[0] << ", " << params.r0[1] << ", " << params.r0[2] << "] km" << std::endl;
        std::cout << "Initial Velocity (v0): [" << params.v0[0] << ", " << params.v0[1] << ", " << params.v0[2] << "] km/s" << std::endl;
        std::cout << "Cross-sectional Area (A): " << params.A << " mÂ²" << std::endl;
        std::cout << "Mass (m): " << params.m << " kg" << std::endl;
        std::cout << "Drag Coefficient (C_D): " << params.C_D << std::endl;
        std::cout << "Num Segments: " << params.num_segments << std::endl;
        std::cout << "Num Gauss-Lobatto Points: " << params.num_gauss_lobatto_points << std::endl;
        std::cout << "=============================" << std::endl
                  << std::endl;

        return true;
    }
    catch (const std::invalid_argument &e)
    {
        std::cerr << "Error: Invalid argument format. Please ensure all parameters are valid numbers." << std::endl;
        std::cerr << "Error details: " << e.what() << std::endl;
        return false;
    }
    catch (const std::out_of_range &e)
    {
        std::cerr << "Error: Number out of range. Please check your input values." << std::endl;
        std::cerr << "Error details: " << e.what() << std::endl;
        return false;
    }
}

int main(int argc, char *argv[])
{
    // Parse command line arguments
    SimulationParams params;
    if (!parseCommandLineArgs(argc, argv, params))
    {
        return 1;
    }

    // Extract parameters for easier use
    const std::string satellite_name = params.satellite_name;
    const auto &r0 = params.r0;
    const auto &v0 = params.v0;
    const double A = params.A * 1e-6; // Convert from m^2 to km^2
    const double m = params.m;
    const double C_D = params.C_D;
    const int num_segments = params.num_segments;
    const int n_points = params.num_gauss_lobatto_points;
    const double total_time = params.total_time;

    // Compute the orbital period
    double orbital_period = compute_satellite_orbital_period(r0, v0);
    std::cout << "Computed Orbital Period: " << orbital_period << " seconds" << std::endl;

    // Define simulation constants
    const double tol = 1e-9;
    const double hmax = orbital_period / num_segments / n_points;
    const double hmin = 1e-14;
    const double RTOL = 1e-9;
    const double ATOL = 1e-12;

    // File to save execution times
    std::string exec_time_filename = "results/"+satellite_name + "_" + 
                               std::to_string(num_segments) + "_" + 
                               std::to_string(n_points) + "_execution_times.csv";
    std::ofstream exec_time_file(exec_time_filename);
    if (!exec_time_file.is_open())
    {
        std::cerr << "Error: Could not open file to save execution times.\n";
        return 1;
    }
    exec_time_file << "Algorithm,Tspan Generation (s),Integration Time (s),Total Time (s)\n";

    // Generate the full Gauss-Lobatto time span
    auto tspan_start = std::chrono::high_resolution_clock::now();
    std::vector<double> full_tspan = generate_full_gauss_lobatto_tspan(total_time, orbital_period, num_segments, n_points);
    auto tspan_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> tspan_elapsed = tspan_end - tspan_start;
    std::cout << "Tspan generation time: " << tspan_elapsed.count() << " seconds\n";

    // Combine initial position and velocity into y0
    std::vector<double> y0 = r0;
    y0.insert(y0.end(), v0.begin(), v0.end());

    // RK4
    auto start = std::chrono::high_resolution_clock::now();
    auto rk4_results = RK4(a_c_func_new, full_tspan, y0, A, m, C_D);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::vector<std::vector<double>> rk4_positions;
    for (const auto &state : rk4_results)
    {
        rk4_positions.push_back({state[0], state[1], state[2]});
    }
    save_final_positions_to_csv(satellite_name, "RK4", full_tspan, rk4_positions, num_segments, n_points);
    exec_time_file << "RK4," << tspan_elapsed.count() << "," << elapsed.count() << "," << (tspan_elapsed.count() + elapsed.count()) << "\n";
    std::cout << "RK4 integration time: " << elapsed.count() << " seconds (Total with tspan: " << (tspan_elapsed.count() + elapsed.count()) << " seconds)\n";

    // RK8
    start = std::chrono::high_resolution_clock::now();
    auto rk8_results = RK8(a_c_func_new, full_tspan, y0, A, m, C_D);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::vector<std::vector<double>> rk8_positions;
    for (const auto &state : rk8_results)
    {
        rk8_positions.push_back({state[0], state[1], state[2]});
    }
    save_final_positions_to_csv(satellite_name, "RK8", full_tspan, rk8_positions, num_segments, n_points);
    exec_time_file << "RK8," << tspan_elapsed.count() << "," << elapsed.count() << "," << (tspan_elapsed.count() + elapsed.count()) << "\n";
    std::cout << "RK8 integration time: " << elapsed.count() << " seconds (Total with tspan: " << (tspan_elapsed.count() + elapsed.count()) << " seconds)\n";

    // ODE45
    start = std::chrono::high_resolution_clock::now();
    auto ode45_results = ode45(a_c_func_new, full_tspan, y0, A, m, C_D, RTOL, ATOL);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::vector<std::vector<double>> ode45_positions;
    for (const auto &state : ode45_results)
    {
        ode45_positions.push_back({state[0], state[1], state[2]});
    }
    save_final_positions_to_csv(satellite_name, "ODE45", full_tspan, ode45_positions, num_segments, n_points);
    exec_time_file << "ODE45," << tspan_elapsed.count() << "," << elapsed.count() << "," << (tspan_elapsed.count() + elapsed.count()) << "\n";
    std::cout << "ODE45 integration time: " << elapsed.count() << " seconds (Total with tspan: " << (tspan_elapsed.count() + elapsed.count()) << " seconds)\n";

    // ODE78
    start = std::chrono::high_resolution_clock::now();
    // b, bh are global, preparing a c_vector, a for ODE78

    std::vector<double> c_vector(c.size());
    for (const auto &pair : c)
    {
        c_vector[pair.first - 1] = pair.second; // Convert map to vector, adjusting index
    }

    auto ode78_results = ode78(a_c_func_new, full_tspan, y0, b, bh, c_vector, a, RTOL, ATOL, A, m, C_D);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::vector<std::vector<double>> ode78_positions;
    for (const auto &state : ode78_results)
    {
        ode78_positions.push_back({state[0], state[1], state[2]});
    }
    save_final_positions_to_csv(satellite_name, "ODE78", full_tspan, ode78_positions, num_segments, n_points);
    exec_time_file << "ODE78," << tspan_elapsed.count() << "," << elapsed.count() << "," << (tspan_elapsed.count() + elapsed.count()) << "\n";
    std::cout << "ODE78 integration time: " << elapsed.count() << " seconds (Total with tspan: " << (tspan_elapsed.count() + elapsed.count()) << " seconds)\n";

    // ODE113
    start = std::chrono::high_resolution_clock::now();
    auto ode113_results = ODE113(a_c_func_new, full_tspan, y0, RTOL, ATOL, hmax, hmin, A, m, C_D);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::vector<std::vector<double>> ode113_positions;
    for (const auto &state : ode113_results){
        ode113_positions.push_back({state[0], state[1], state[2]});
    }
    save_final_positions_to_csv(satellite_name, "ODE113", full_tspan, ode113_positions, num_segments, n_points);
    exec_time_file << "ODE113," << tspan_elapsed.count() << "," << elapsed.count() << "," << (tspan_elapsed.count() + elapsed.count()) << "\n";
    std::cout << "ODE113 integration time: " << elapsed.count() << " seconds (Total with tspan: " << (tspan_elapsed.count() + elapsed.count()) << " seconds)\n";

    // MPCI
    start = std::chrono::high_resolution_clock::now();
    std::vector<double> tspan_simple = {0.0, total_time}; // Just [start, end]
    auto mpci_results = MPCI(a_c_func_new, tspan_simple, y0, 1e-12, 1e-9, orbital_period / num_segments, 2000, n_points, A, m, C_D);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::vector<std::vector<double>> mpci_positions;
    for (const auto &state : mpci_results)
    {
        mpci_positions.push_back({state[0], state[1], state[2]});
    }
    save_final_positions_to_csv(satellite_name, "MPCI", full_tspan, mpci_positions, num_segments, n_points);
    exec_time_file << "MPCI,0.0," << elapsed.count() << "," << elapsed.count() << "\n";
    std::cout << "MPCI total time: " << elapsed.count() << " seconds (includes internal time point generation)\n";

    // Close the execution time file
    exec_time_file.close();

    return 0;
}
