// RK4.h
#ifndef RK4_H
#define RK4_H

#include <vector>

std::vector<std::vector<double>> RK4(
    std::vector<double>(*odefun)(double, const std::vector<double>&, double, double, double),
    const std::vector<double>& t_gauss_lobatto,
    const std::vector<double>& y0
);

#endif // RK4_H