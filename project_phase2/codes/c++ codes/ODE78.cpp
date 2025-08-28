#include <iostream>
#include <vector>
#include <cmath>
#include <tuple>
#include <algorithm>
#include <functional> // Required for std::function
#include <map>
#include <iomanip>
#include <numeric> // for std::accumulate
#include "CommonFunctions.h"
#include "coefficients78.h"

// RK78 single step function
std::tuple<double, std::vector<double>, double> rk78_step(
    std::function<std::vector<double>(double, const std::vector<double> &, double, double, double)> ode_func,
    double t,
    const std::vector<double> &y,
    double h,
    double rtol,
    double atol,
    const std::vector<double> &b,                  // 8th-order weights
    const std::vector<double> &bh,                 // 7th-order weights
    const std::vector<double> &c,                  // Time nodes (Gauss-Lobatto points)
    const std::map<int, std::map<int, double>> &a, // Coupling coefficients (Butcher tableau)
    double A,
    double m,
    double C_D)
{
    // Initialize stages (k1 to k13)
    std::vector<std::vector<double>> k(13, std::vector<double>(y.size()));

    k[0] = scalar_multiply(h, ode_func(t, y, A, m, C_D));

    for (int i = 1; i < 13; ++i)
    {
        std::vector<double> y_temp = y;
        for (int j = 0; j < i; ++j)
        {
            auto it = a.find(i + 1); // Find the ith row in the map
            if (it != a.end())
            {                                           // Check if this row exists
                auto it_inner = it->second.find(j + 1); // Find the element in the inner map
                if (it_inner != it->second.end())
                { // Check if the inner element exists
                    y_temp = vector_add(y_temp, scalar_multiply(it_inner->second, k[j]));
                }
            }
        }
        k[i] = scalar_multiply(h, ode_func(t + c[i + 1] * h, y_temp, A, m, C_D));
    }

    // Compute the 8th-order solution (y8) using the b coefficients
    std::vector<double> y8 = vector_add(y, dot_product(b, k));

    // Compute the 7th-order solution (y7) using the bh coefficients
    std::vector<double> y7 = vector_add(y, dot_product(bh, k));

    // Estimate the error between the 7th and 8th order solutions
    double error = 0.0;
    for (size_t i = 0; i < y8.size(); ++i)
    {
        error = std::max(error, std::abs(y8[i] - y7[i]) / (atol + rtol * std::max(std::abs(y7[i]), std::abs(y8[i]))));
    }

    // Adjust step size based on the error estimate
    double h_new = (error != 0) ? h * std::min(2.0, std::max(0.1, 0.9 / std::pow(error, 0.2))) : h * 2;

    return std::make_tuple(t + h, y8, h_new);
}

// RK78 integration function
std::vector<std::vector<double>> ode78(
    std::function<std::vector<double>(double, const std::vector<double> &, double, double, double)> ode_func,
    const std::vector<double> &t_span,
    const std::vector<double> &y0,
    const std::vector<double> &b,                  // 8th-order weights
    const std::vector<double> &bh,                 // 7th-order weights
    const std::vector<double> &c,                  // Time nodes (Gauss-Lobatto points)
    const std::map<int, std::map<int, double>> &a, // Coupling coefficients (Butcher tableau)
    double rtol = 1e-3,
    double atol = 1e-6,
    double A = 4.0,
    double m = 2.4,
    double C_D = 2.2)
{
    //std::vector<double> tout = {t_span[0]};
    std::vector<std::vector<double>> yout = {y0};

    double t = t_span[0];
    std::vector<double> y = y0;

    for (size_t i = 1; i < t_span.size(); ++i)
    {
        double h = t_span[i] - t_span[i - 1];
        while (t < t_span[i])
        {
            if (t + h > t_span[i])
            {
                h = t_span[i] - t;
            }
            auto [t_next, y_next, h_new] = rk78_step(ode_func, t, y, h, rtol, atol, b, bh, c, a, A, m, C_D);
            t = t_next;
            y = y_next;
            h = h_new;
        }
        if (abs(t - t_span[i]) > 1e-12)
        {
            std::cerr << "Warning: ODE78 landed at t=" << std::setprecision(15) << t
                      << " instead of target t_span[" << i << "]=" << t_span[i]
                      << " (difference: " << (t - t_span[i]) << ")" << std::endl;
        }
        //tout.push_back(t);
        yout.push_back(y);
    }

    return yout;
}

/*
int main() {
    double tspan_start = 0;
    double tspan_end = 1000000;
    double mu = 398600; // Gravitational parameter
    int n_points = 32;

    // Initial position and velocity vectors
    std::vector<double> r0 = {-3314.58372, 2809.02481, -5400.46222};
    std::vector<double> v0 = {-3.42701791, -6.62341508, -1.34238849};
    std::vector<double> y0 = r0;
    y0.insert(y0.end(), v0.begin(), v0.end());

    // Load Gauss-Lobatto points for the time span
    auto gauss_lobatto_tspan = gaussLobattoPoints(n_points, tspan_start, tspan_end);

    // Convert c from map<int, double> to vector<double>
    std::vector<double> c_vector(c.size());
    for (const auto& pair : c) {
        c_vector[pair.first - 1] = pair.second;  // Convert map to vector, adjusting index
    }

    // Coefficients b, bh, c, and a are already defined globally in the code
    // No need to redefine them here, just use the global definitions

    auto func = [&mu](double t, const std::vector<double>& y) { return satellite_motion(t, y, mu); };

    // Run RK78 with Gauss-Lobatto points
    auto result = ode78(func, gauss_lobatto_tspan, y0, b, bh, c_vector, a, 1e-8, 1e-10);

    // Print results
    std::cout << "Results (Time | Position X, Y, Z | Velocity X, Y, Z):\n";
    std::cout << "Time      PosX       PosY       PosZ       VelX       VelY       VelZ\n";
    std::cout << "--------------------------------------------------------------\n";
    for (const auto& row : result) {
        for (double value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
*/