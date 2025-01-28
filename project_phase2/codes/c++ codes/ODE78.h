// ODE78.h
#ifndef ODE78_H
#define ODE78_H

#include <vector>
#include <sstream>
#include <functional>
#include <map>

// ODE78 function declaration
std::vector<std::vector<double>> ode78(
    std::function<std::vector<double>(double, const std::vector<double>&,double,double,double)> ode_func,
    const std::vector<double>& t_span,
    const std::vector<double>& y0,
    const std::vector<double>& b,  // 8th-order weights
    const std::vector<double>& bh, // 7th-order weights
    const std::vector<double>& c,  // Time nodes (Gauss-Lobatto points)
    const std::map<int, std::map<int, double>>& a,  // Coupling coefficients (Butcher tableau)
    double rtol = 1e-3,
    double atol = 1e-6,
    double A=12,
    double m=2000,
    double C_D=2.2
);

#endif // ODE78_H
