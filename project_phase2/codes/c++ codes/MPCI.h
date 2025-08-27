#ifndef MPCI_H
#define MPCI_H

#include <Eigen/Dense>  // This gives you matrices, vectors, etc.
#include <vector>
#include <functional>

// MPCI propagator function
std::vector<std::vector<double>> MPCI(
    std::vector<double>(*odefun)(double, const std::vector<double>&, double, double, double),
    const std::vector<double>& tspan,
    const std::vector<double>& x0,
    double abs_tol = 1e-12,
    double rel_tol = 1e-9,
    double Sec = -1,  // in satellites Sec is the Orbital period/num_segments. -1 means use tspan.back()
    int MaxIter=2000,
    int N=16,
    double A=3.9,
    double m=260,
    double C_D=2.2
);
#endif // MPCI_H