// SatelliteUtils.h
#ifndef SATELLITE_UTILS_H
#define SATELLITE_UTILS_H

#include <vector>
#include <cmath>
#include <string> // Include this to use std::string
#include <tuple>  // Include this to use std::tuple
#include <array>

const double mu = 398600.4418;  // Standard gravitational parameter for Earth
// Constants
extern const double J2;
extern const double R_E;
extern const double MU;

// Function to calculate the norm of a vector
double vectorNorm(const std::vector<double>& vec);

// Function for vector addition
std::vector<double> vectorAdd(const std::vector<double>& a, const std::vector<double>& b);

// Function for scalar multiplication of a vector
std::vector<double> scalarMultiply(const std::vector<double>& vec, double scalar);

// Function to calculate satellite motion given time t and state vector y
std::vector<double> satelliteMotion(double t, const std::vector<double>& y);

// Function to generate Gauss-Lobatto points in a given range
std::vector<double> gaussLobattoPoints(int n, double a, double b);

std::vector<double> generate_full_gauss_lobatto_tspan(double total_time, double orbital_period, int num_segments, int n_points);

// Satellite motion model differential equation
std::vector<double> satellite_motion(double t, const std::vector<double>& y, double mu);

// Overload operators for vector and scalar arithmetic
std::vector<double> operator+(const std::vector<double>& a, const std::vector<double>& b);
std::vector<double> operator+(double scalar, const std::vector<double>& v);
std::vector<double> operator+(const std::vector<double>& v, double scalar);

std::vector<double> operator-(const std::vector<double>& a, const std::vector<double>& b);
std::vector<double> operator-(double scalar, const std::vector<double>& v);
std::vector<double> operator-(const std::vector<double>& v, double scalar);

std::vector<double> operator*(const std::vector<double>& v, double scalar);
std::vector<double> operator*(double scalar, const std::vector<double>& v);

std::vector<double> operator/(const std::vector<double>& v, double scalar);

// Helper function for dot product
std::vector<double> dot_product(const std::vector<double>& weights, const std::vector<std::vector<double>>& k);

// Helper function to scale a vector by a scalar
std::vector<double> scalar_multiply(double scalar, const std::vector<double>& vec);

// Norm function to calculate vector norm
double vector_norm(const std::vector<double>& vec);

// Helper function to perform element-wise vector addition
std::vector<double> vector_add(const std::vector<double>& v1, const std::vector<double>& v2);

// Atmospheric density
double atmospheric_density(double altitude);

// Advanced satellite dynamics model
std::vector<double> a_c_func(double t, const std::vector<double>& y, double A, double m, double C_D = 2.2);

std::vector<double> a_c_func_new(double t, const std::vector<double>& y, double A, double m, double C_D );
// Satellite parameter retrieval
std::tuple<double, double, double> get_satellite_params(const std::string& sat_name);

// Function to calculate the norm of a vector
double vectorNorm(const std::array<double, 3>& vec);

// Atmospheric density model based on altitude
double atmosphericDensity(double altitude);

//Compute the satellite orbital period
double compute_satellite_orbital_period( const std::vector<double>& r0,const std::vector<double>& v0);


#endif // SATELLITE_UTILS_H
