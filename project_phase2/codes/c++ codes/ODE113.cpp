#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include "CommonFunctions.h"

using namespace std;

// ODE113 implementation
std::vector<std::vector<double>> ODE113(
    std::vector<double> (*ode)(double, const std::vector<double> &, double, double, double),
    const vector<double> &tspan,
    const vector<double> &y0,
    double rel_tol = 1e-9,
    double abs_tol = 1e-9,
    double hmax = 10.0,
    double hmin = 0.001,
    double A = 12,
    double m = 2000,
    double C_D = 2.2)
{
    std::vector<std::vector<double>>  yout;
    double t = tspan[0];
    double t_end = tspan.back();
    double h = 0.01; // Initial step size
    vector<double> y = y0;

    yout.push_back(y);

    size_t tspan_index = 1;

    for (size_t i = 1; i < tspan.size(); ++i)
    {
        double target_time = tspan[i];

        while (t < target_time)
        {
            // Check for overshoot
            if (t + h > target_time)
            {
                h = target_time - t;
            }

            // Compute derivative
            vector<double> yp = ode(t, y, A, m, C_D);
            vector<double> y_pred(y.size()), y_corr(y.size()), yp_corr(y.size());

            // Predictor step
            for (size_t j = 0; j < y.size(); j++)
            {
                y_pred[j] = y[j] + h * yp[j];
            }

            // STORE the initial predictor for error estimation
            //vector<double> y_pred_initial = y_pred;
            vector<double> y_prev_corr;

            // Corrector step
            for (int iter = 0; iter < 3; iter++)
            {
                yp_corr = ode(t + h, y_pred, A, m, C_D);

                // SAVE the previous corrector BEFORE computing the new one
                if (iter == 2) {  // Save the result from iteration 1 (second corrector)
                    y_prev_corr = y_pred;  // This contains the corrector from iteration 1
                }

                for (size_t j = 0; j < y.size(); j++)
                {
                    y_corr[j] = y[j] + h * (yp[j] + yp_corr[j]) / 2.0;
                }
                y_pred = y_corr;
            }

            // Error estimation
            double scaled_error = 0.0;
            for (size_t j = 0; j < y.size(); j++)
            {
                double raw_error = fabs(y_corr[j] - y_prev_corr[j]);
                double scale = abs_tol + rel_tol * max(fabs(y_corr[j]), fabs(y_prev_corr[j]));
                double local_scaled_error = raw_error / scale;
                scaled_error = max(scaled_error, local_scaled_error);
            }

            if (scaled_error <= 1.0)
            {
                // Accept step
                t += h;
                y = y_corr;

                // Adjust step size for next iteration (only if not at target)
                if (t < target_time)
                {
                    double safety_factor = 0.9;
                    double power = 0.5;
                    double eps_min = 1e-15;
                    double factor = safety_factor * pow(1.0 / max(scaled_error, eps_min), power);
                    factor = max(0.1, min(2.0, factor));
                    h = max(min(h * factor, hmax), hmin);
                }
            }
            else
            {
                // Reject step and reduce step size
                double safety_factor = 0.9;
                double power = 0.5;
                double eps_min = 1e-15;
                double factor = safety_factor * pow(1.0 / max(scaled_error, eps_min), power);
                factor = max(0.1, min(2.0, factor));
                h = max(min(h * factor, hmax), hmin);
            }

            if (h < hmin)
            {
                h = hmin;
            }
        }

        // Safety check - warn if we're not exactly on target
        if (abs(t - target_time) > 1e-12)
        {
            std::cerr << "Warning: ODE113 landed at t=" << std::setprecision(15) << t
                      << " instead of target t_span[" << i << "]=" << target_time
                      << " (difference: " << (t - target_time) << ")" << std::endl;
        }

        yout.push_back(y);
    }

    return yout;
}
