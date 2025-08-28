// RK4.h
#ifndef RK4_H
#define RK4_H

#include <vector>

std::vector<std::vector<double>> RK4(
    std::vector<double>(*odefun)(double, const std::vector<double>&, double, double, double), // Updated signature
    const std::vector<double>& t_gauss_lobatto,
    const std::vector<double>& y0,
    double A,  // Cross-sectional area
    double m,  // Satellite mass
    double C_D // Drag coefficient
);

#endif // RK4_H