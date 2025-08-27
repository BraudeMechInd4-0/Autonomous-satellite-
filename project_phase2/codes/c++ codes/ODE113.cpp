#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include "CommonFunctions.h"

using namespace std;

struct Result {
    vector<double> time;
    vector<vector<double>> values;
};



// ODE113 implementation
Result ODE113(
    std::vector<double> (*ode)(double, const std::vector<double>&, double,double,double),
    const vector<double>& tspan,
    const vector<double>& y0,
    double mu,
    double rel_tol = 1e-9,
    double abs_tol = 1e-9,
    double hmax = 10.0,
    double hmin = 0.001,
    double A=12,
    double m=2000,
    double C_D=2.2
) {
    Result result;
    double t = tspan[0];
    double t_end = tspan.back();
    double h = 0.01; // Initial step size
    vector<double> y = y0;

    result.time.push_back(t);
    result.values.push_back(y);

    size_t tspan_index = 1;

    while (t < t_end) {
        if (tspan_index < tspan.size() && t + h > tspan[tspan_index]) {
            h = tspan[tspan_index] - t;
        }

        vector<double> yp = ode(t, y, A,m,C_D);
        vector<double> y_pred(y.size()), y_corr(y.size()), yp_corr(y.size());

        // Predictor step
        for (size_t i = 0; i < y.size(); i++) {
            y_pred[i] = y[i] + h * yp[i];
        }

        // Corrector step
        for (int iter = 0; iter < 3; iter++) {
            yp_corr = ode(t + h, y_pred, A,m,C_D);
            for (size_t i = 0; i < y.size(); i++) {
                y_corr[i] = y[i] + h * (yp[i] + yp_corr[i]) / 2.0;
            }
            y_pred = y_corr;
        }

        // Error estimation
        double scaled_error = 0.0;
        for (size_t i = 0; i < y.size(); i++) {
            double raw_error = fabs(y_corr[i] - y_pred[i]);
            double scale = abs_tol + rel_tol * max(fabs(y_corr[i]), fabs(y_pred[i]));
            double local_scaled_error = raw_error / scale;
            scaled_error = max(scaled_error, local_scaled_error);
        }

        if (scaled_error <= 1.0) {
            // Accept step
            t += h;
            y = y_corr;
            result.time.push_back(t);
            result.values.push_back(y);

            // Check if we've reached a tspan point
            if (tspan_index < tspan.size() && t >= tspan[tspan_index] - 1e-12) {
                tspan_index++;
            }
        }

        // Adjust step size
        double safety_factor = 0.9;
        double power = 0.5;  // Appropriate for predictor-corrector methods
        double eps_min = 1e-15;  // Prevent division by zero
        
        double factor = safety_factor * pow(1.0 / max(scaled_error, eps_min), power);
        factor = max(0.1, min(2.0, factor));  // Limit step size changes
        h = max(min(h * factor, hmax), hmin);

        if (h < hmin) {
            cerr << "Warning: Step size below minimum. Exiting loop." << endl;
            break;
        }
    }

    return result;
}


