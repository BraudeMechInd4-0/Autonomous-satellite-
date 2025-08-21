#pragma once
#include <vector>

// MPCI propagator function
std::vector<std::vector<double>> MPCI(
    const std::vector<double>& t_span,
    const std::vector<double>& y0,
    int degree,
    int num_segments
);