// SatelliteUtils.cpp
#include "CommonFunctions.h"
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <array>
#include <vector>
#include <iostream>
#include <algorithm>

// Constants
const double J2 = 1.08263e-3;       // Earth's J2 coefficient
const double R_E = 6378.137;        // Earth's radius in km
const double MU = 398600.4418;      // Earth's gravitational parameter (km^3/s^2)
const double PI = 3.1415926535;

double vectorNorm(const std::vector<double>& vec) {
    double sum = 0;
    for (double val : vec) {
        sum += val * val;
    }
    return std::sqrt(sum);
}

std::vector<double> vectorAdd(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

std::vector<double> scalarMultiply(const std::vector<double>& vec, double scalar) {
    std::vector<double> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = vec[i] * scalar;
    }
    return result;
}

std::vector<double> satelliteMotion(double t, const std::vector<double>& y) {
    std::vector<double> r = {y[0], y[1], y[2]};  // Position vector
    std::vector<double> v = {y[3], y[4], y[5]};  // Velocity vector

    double r_norm = vectorNorm(r);  // Magnitude of the position vector

    // Acceleration vector
    std::vector<double> a = {-mu / std::pow(r_norm, 3) * r[0],
                             -mu / std::pow(r_norm, 3) * r[1],
                             -mu / std::pow(r_norm, 3) * r[2]};

    std::vector<double> dydt(6);
    dydt[0] = v[0]; dydt[1] = v[1]; dydt[2] = v[2];  // Velocity components
    dydt[3] = a[0]; dydt[4] = a[1]; dydt[5] = a[2];  // Acceleration components

    return dydt;
}

std::vector<double> gaussLobattoPoints(int n, double a, double b) {
    std::vector<double> points(n);
    for (int i = 0; i < n; ++i) {
        double theta = M_PI * i / (n - 1);
        points[i] = -cos(theta);
    }
    // Scale points to the interval [a, b]
    for (int i = 0; i < n; ++i) {
        points[i] = 0.5 * (b - a) * (points[i] + 1) + a;
    }
    return points;
}

std::vector<double> generate_full_gauss_lobatto_tspan(double total_time, double orbital_period, int num_segments, int n_points) {
    std::vector<double> full_tspan;
    double segment_duration = orbital_period / num_segments;
    std::cout << "Segment Duration: " << segment_duration << " seconds" << std::endl;
    int total_segments = static_cast<int>(std::ceil(total_time / segment_duration));
    std::cout << "Total Segments: " << total_segments << std::endl;

    for (int segment = 0; segment < total_segments; ++segment) {
        double t_start = segment * segment_duration;
        double t_end = std::min(t_start + segment_duration, total_time);  // Handle partial segment at end
        
        std::vector<double> segment_points = gaussLobattoPoints(n_points, t_start, t_end);
        
        if (segment == 0) {
            // First segment: include all points
            full_tspan.insert(full_tspan.end(), segment_points.begin(), segment_points.end());
        } else {
            // Subsequent segments: skip first point (it's the same as last point of previous segment)
            full_tspan.insert(full_tspan.end(), segment_points.begin() + 1, segment_points.end());
        }
    }
    
    return full_tspan;
}

// Overload the + operator for vector addition (vector + vector)
std::vector<double> operator+(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of the same length");
    }
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

// Overload the + operator for scalar and vector addition (scalar + vector)
std::vector<double> operator+(double scalar, const std::vector<double>& v) {
    std::vector<double> result(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = scalar + v[i];
    }
    return result;
}

// Overload the + operator for vector and scalar addition (vector + scalar)
std::vector<double> operator+(const std::vector<double>& v, double scalar) {
    return scalar + v;  // Use the scalar + vector overload
}

// Overload the - operator for vector subtraction (vector - vector)
std::vector<double> operator-(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of the same length");
    }
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

// Overload the - operator for scalar and vector subtraction (scalar - vector)
std::vector<double> operator-(double scalar, const std::vector<double>& v) {
    std::vector<double> result(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = scalar - v[i];
    }
    return result;
}

// Overload the - operator for vector and scalar subtraction (vector - scalar)
std::vector<double> operator-(const std::vector<double>& v, double scalar) {
    std::vector<double> result(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = v[i] - scalar;
    }
    return result;
}


// Overload the * operator for scalar multiplication (vector * scalar)
std::vector<double> operator*(const std::vector<double>& v, double scalar) {
    std::vector<double> result(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = v[i] * scalar;
    }
    return result;
}

// Overload the * operator for scalar multiplication (scalar * vector)
std::vector<double> operator*(double scalar, const std::vector<double>& v) {
    return v * scalar;  // Use the other overloaded operator
}

// Overload the / operator for vector and scalar division (vector / scalar)
std::vector<double> operator/(const std::vector<double>& v, double scalar) {
    if (scalar == 0.0) {
        throw std::invalid_argument("Division by zero");
    }
    std::vector<double> result(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = v[i] / scalar;
    }
    return result;
}

// Function to calculate satellite motion
std::vector<double> satelliteMotion(double t, const std::vector<double>& y, double mu) {
    std::vector<double> r(y.begin(), y.begin() + 3);  // Position vector
    std::vector<double> v(y.begin() + 3, y.end());    // Velocity vector

    double r_norm = vectorNorm(r);  // Magnitude of position vector

    // Acceleration vector
    std::vector<double> a(3);
    for (int i = 0; i < 3; ++i) {
        a[i] = -mu / std::pow(r_norm, 3) * r[i];
    }

    // Combine velocity and acceleration into a single vector
    std::vector<double> dydt(6);
    for (int i = 0; i < 3; ++i) {
        dydt[i] = v[i];     // Velocity
        dydt[i + 3] = a[i]; // Acceleration
    }

    return dydt;
}

// Satellite motion model differential equation
std::vector<double> satellite_motion(double t, const std::vector<double>& y, double mu) {
    std::vector<double> r(y.begin(), y.begin() + 3);  // Position vector
    std::vector<double> v(y.begin() + 3, y.end());    // Velocity vector
    double r_norm = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);

    // Acceleration
    std::vector<double> a(3);
    for (int i = 0; i < 3; ++i) {
        a[i] = -mu / (r_norm * r_norm * r_norm) * r[i];
    }

    // Combine velocity and acceleration
    std::vector<double> dydt(6);
    for (int i = 0; i < 3; ++i) {
        dydt[i] = v[i];  // Velocity components
        dydt[i + 3] = a[i];  // Acceleration components
    }

    return dydt;
}

// Helper function for dot product (replacing the previous approach)
std::vector<double> dot_product(const std::vector<double>& weights, const std::vector<std::vector<double>>& k) {
    std::vector<double> result(k[0].size(), 0.0);
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < result.size(); ++j) {
            result[j] += weights[i] * k[i][j];
        }
    }
    return result;
}

// Helper function to scale a vector by a scalar
std::vector<double> scalar_multiply(double scalar, const std::vector<double>& vec) {
    std::vector<double> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = scalar * vec[i];
    }
    return result;
}

// Norm function to calculate vector norm (like np.linalg.norm in Python)
double vector_norm(const std::vector<double>& vec) {
    return std::sqrt(std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0));
}
// Helper function to perform element-wise vector addition
std::vector<double> vector_add(const std::vector<double>& v1, const std::vector<double>& v2) {
    std::vector<double> result(v1.size());
    for (size_t i = 0; i < v1.size(); ++i) {
        result[i] = v1[i] + v2[i];
    }
    return result;
}

#include <cmath>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

// Function to compute atmospheric density based on altitude
double atmospheric_density(double altitude) {
    struct AltitudeData {
        double h0, rho0, H;
    };

    std::vector<AltitudeData> altitude_data = {
            {0, 1.225, 7.249}, {25, 3.899e-2, 6.349}, {30, 1.774e-2, 6.682},
            {40, 3.972e-3, 7.554}, {50, 1.057e-3, 8.382}, {60, 3.206e-4, 7.714},
            {70, 8.77e-5, 6.549}, {80, 1.905e-5, 5.799}, {90, 3.396e-6, 5.382},
            {100, 5.297e-7, 5.877}, {110, 9.661e-8, 7.263}, {120, 2.438e-8, 9.473},
            {130, 8.484e-9, 12.636}, {140, 3.845e-9, 16.149}, {150, 2.07e-9, 22.523},
            {180, 5.464e-10, 29.74}, {200, 2.789e-10, 37.105}, {250, 7.248e-11, 45.546},
            {300, 2.418e-11, 53.628}, {350, 9.518e-12, 53.298}, {400, 3.725e-12, 58.515},
            {450, 1.585e-12, 60.828}, {500, 6.967e-13, 63.822}, {600, 1.454e-13, 71.835},
            {700, 3.614e-14, 88.667}, {800, 1.17e-14, 124.64}, {900, 5.245e-15, 181.05},
            {1000, 3.019e-15, 268.00}
    };

    for (size_t i = 0; i < altitude_data.size() - 1; ++i) {
        const auto& current = altitude_data[i];
        const auto& next = altitude_data[i + 1];
        if (current.h0 <= altitude && altitude < next.h0) {
            return current.rho0 * std::exp(-(altitude - current.h0) / current.H);
        }
    }

    return altitude_data.back().rho0;
}

std::vector<double> a_c_func(double t, const std::vector<double>& y,double A,double m,double C_D) {

    std::vector<double> r(y.begin(), y.begin() + 3);  // Position vector [x, y, z]
    std::vector<double> v(y.begin() + 3, y.end());    // Velocity vector [v_x, v_y, v_z]

    //const double R_E = 6378.137;

    double r_norm = std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    double altitude = r_norm - R_E;
    double rho = atmospheric_density(altitude);

    double x = r[0], y_pos = r[1], z = r[2];  // Decompose the position vector

    // J2 acceleration components
    double factor = -(3.0 / 2.0) * J2 * mu * (R_E * R_E) / pow(r_norm, 5);
    double z2_r2 = (z / r_norm) * (z / r_norm);  // (z / r)^2 term


    std::vector<double> a_c(3);
    a_c[0] = factor * x * (1 - 5 * z2_r2);
    a_c[1] = factor * y_pos * (1 - 5 * z2_r2);
    a_c[2] = factor * z * (3 - 5 * z2_r2);

    // Drag acceleration components
    double v_rel_norm = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);  // Relative velocity magnitude
    double drag_factor = -0.5 * C_D * A / m * rho;  // Use constant rho
    std::vector<double> a_d(3);
    a_d[0] = drag_factor * v_rel_norm * v[0];
    a_d[1] = drag_factor * v_rel_norm * v[1];
    a_d[2] = drag_factor * v_rel_norm * v[2];

    // Gravity acceleration components
    std::vector<double> a_g(3);
    a_g[0] = (-mu / pow(r_norm, 3)) * x;
    a_g[1] = (-mu / pow(r_norm, 3)) * y_pos;
    a_g[2] = (-mu / pow(r_norm, 3)) * z;

    // Total acceleration
    std::vector<double> total_acceleration(3);
    total_acceleration[0] = a_g[0] + a_c[0] + a_d[0];
    total_acceleration[1] = a_g[1] + a_c[1] + a_d[1];
    total_acceleration[2] = a_g[2] + a_c[2] + a_d[2];

    // Derivatives
    std::vector<double> dy(6);
    dy[0] = v[0];  // dx/dt
    dy[1] = v[1];  // dy/dt
    dy[2] = v[2];  // dz/dt
    dy[3] = total_acceleration[0];  // dv_x/dt
    dy[4] = total_acceleration[1];  // dv_y/dt
    dy[5] = total_acceleration[2];  // dv_z/dt

    return dy;
}

// Function to calculate satellite dynamics
std::vector<double> a_c_func_new(
        double t, const std::vector<double>& y, double A, double m, double C_D ) {

    std::vector<double> r(y.begin(), y.begin() + 3);  // Position vector [x, y, z]
    std::vector<double> v(y.begin() + 3, y.end());    // Velocity vector [v_x, v_y, v_z]
    //const double mu = 398600.4418;
    //const double R_E = 6378.137;

    double r_norm = std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    double altitude = r_norm - R_E;
    double rho = atmospheric_density(altitude);

    double x = r[0], y_pos = r[1], z = r[2];  // Decompose the position vector

    // J2 acceleration components
    double factor = -(3.0 / 2.0) * J2 * MU * (R_E * R_E) / pow(r_norm, 5);
    double z2_r2 = (z / r_norm) * (z / r_norm);  // (z / r)^2 term

    std::vector<double> a_c(3);
    a_c[0] = factor * x * (1 - 5 * z2_r2);
    a_c[1] = factor * y_pos * (1 - 5 * z2_r2);
    a_c[2] = factor * z * (3 - 5 * z2_r2);

    // Drag acceleration components
    double v_rel_norm = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);  // Relative velocity magnitude
    double drag_factor = -0.5 * C_D * A / m * rho;  // Use constant rho
    std::vector<double> a_d(3);
    a_d[0] = drag_factor * v_rel_norm * v[0];
    a_d[1] = drag_factor * v_rel_norm * v[1];
    a_d[2] = drag_factor * v_rel_norm * v[2];

    // Gravity acceleration components
    std::vector<double> a_g(3);
    a_g[0] = (-MU / pow(r_norm, 3)) * x;
    a_g[1] = (-MU / pow(r_norm, 3)) * y_pos;
    a_g[2] = (-MU / pow(r_norm, 3)) * z;

    // Total acceleration
    std::vector<double> total_acceleration(3);
    total_acceleration[0] = a_g[0] + a_c[0] + a_d[0];
    total_acceleration[1] = a_g[1] + a_c[1] + a_d[1];
    total_acceleration[2] = a_g[2] + a_c[2] + a_d[2];

    // Derivatives
    std::vector<double> dy(6);
    dy[0] = v[0];  // dx/dt
    dy[1] = v[1];  // dy/dt
    dy[2] = v[2];  // dz/dt
    dy[3] = total_acceleration[0];  // dv_x/dt
    dy[4] = total_acceleration[1];  // dv_y/dt
    dy[5] = total_acceleration[2];  // dv_z/dt

    return dy;
}

// Function to get satellite parameters
std::tuple<double, double, double> get_satellite_params(const std::string& sat_name) {
    std::unordered_map<std::string, std::tuple<double, double, double>> satellite_data = {
            {"STARLINK-1341", {3.9, 260, 2.2}},
            {"IRIDIUM 33 DEB", {0.7, 77, 2.2}},
            {"QIANFAN-4", {4.0, 260, 2.2}},
            {"SKYNET 4C", {10.0, 1250, 2.2}},
            {"ASBM-2", {12.0, 2000, 2.2}}
    };

    auto it = satellite_data.find(sat_name);
    if (it != satellite_data.end()) {
        return it->second;
    }

    return {0.0, 0.0, 0.0}; // Default if not found
}



// Helper function to calculate the norm of a vector
double vectorNorm(const std::array<double, 3>& vec) {
    return sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
}

// Atmospheric density model based on altitude
double atmosphericDensity(double altitude) {
    std::vector<std::tuple<double, double, double>> altitude_data = {
        {0, 1.225, 7.249}, {25, 3.899e-2, 6.349}, {30, 1.774e-2, 6.682},
        {40, 3.972e-3, 7.554}, {50, 1.057e-3, 8.382}, {60, 3.206e-4, 7.714},
        {70, 8.77e-5, 6.549}, {80, 1.905e-5, 5.799}, {90, 3.396e-6, 5.382},
        {100, 5.297e-7, 5.877}, {110, 9.661e-8, 7.263}, {120, 2.438e-8, 9.473},
        {130, 8.484e-9, 12.636}, {140, 3.845e-9, 16.149}, {150, 2.070e-9, 22.523},
        {180, 5.464e-10, 29.740}, {200, 2.789e-10, 37.105}, {250, 7.248e-11, 45.546},
        {300, 2.418e-11, 53.628}, {350, 9.518e-12, 53.298}, {400, 3.725e-12, 58.515},
        {450, 1.585e-12, 60.828}, {500, 6.967e-13, 63.822}, {600, 1.454e-13, 71.835},
        {700, 3.614e-14, 88.667}, {800, 1.170e-14, 124.64}, {900, 5.245e-15, 181.05},
        {1000, 3.019e-15, 268.00}
    };

    if (altitude < std::get<0>(altitude_data.front())){
        throw std::out_of_range("Warning: Altitude " + std::to_string(altitude) + " km is below the valid range. Setting density to 0.");
    }

    for (size_t i = 0; i < altitude_data.size() - 1; ++i) {
        auto [h0, rho0, H] = altitude_data[i];
        auto [h1, rho1, _] = altitude_data[i + 1];

        if (altitude >= h0 && altitude < h1) {
            return rho0 * exp(-(altitude - h0) / H);
        }
    }
    return std::get<1>(altitude_data.back());
}


double compute_satellite_orbital_period( const std::vector<double>& r0,const std::vector<double>& v0){
    double r,v,a;
    r = vectorNorm(r0);
    v = vectorNorm(v0);
    a = 1/(2/r-v*v/MU);
    return 2*PI*sqrt(a*a*a/MU);
}

