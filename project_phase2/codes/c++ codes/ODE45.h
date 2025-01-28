// ODE45.h
#ifndef ODE45_H
#define ODE45_H

#include <vector>

// ODE45 function declaration
std::vector<std::vector<double>> ode45(
std::vector<double>(*func)(double, const std::vector<double>&, double, double, double),
    const std::vector<double>& t_span,
    const std::vector<double>& y0,
    double A,  // Cross-sectional area
    double m,  // Mass
    double C_D, // Drag coefficient
    double rtol = 1e-6,
    double atol = 1e-6
    );

#endif // ODE45_H
