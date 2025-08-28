#ifndef ODE113_H
#define ODE113_H

#include <vector>

// ODE113 function declaration
std::vector<std::vector<double>> ODE113(
    std::vector<double> (*ode)(double, const std::vector<double>&, double,double,double), // ODE function
    const std::vector<double>& time_points,  // Time span or specific points
    const std::vector<double>& y0,          // Initial conditions
    double rel_tol = 1e-9,                   // Relative tolerance
    double abs_tol = 1e-9,                   // Absolute tolerance
    double hmax = 1.0,                       // Maximum step size
    double hmin = 1e-6,
    double A = 12,
    double m =2000,
    double C_D=2.2
);

#endif // ODE113_H
