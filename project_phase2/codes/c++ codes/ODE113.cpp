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
    std::vector<double> (*ode)(double, const std::vector<double>&, double,double,double,double),
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

        vector<double> yp = ode(t, y, mu,A,m,C_D);
        vector<double> y_pred(y.size()), y_corr(y.size()), yp_corr(y.size());

        // Predictor step
        for (size_t i = 0; i < y.size(); i++) {
            y_pred[i] = y[i] + h * yp[i];
        }

        // Corrector step
        for (int iter = 0; iter < 3; iter++) {
            yp_corr = ode(t + h, y_pred, mu,A,m,C_D);
            for (size_t i = 0; i < y.size(); i++) {
                y_corr[i] = y[i] + h * (yp[i] + yp_corr[i]) / 2.0;
            }
            y_pred = y_corr;
        }

        // Error estimation
        double err = 0.0;
        for (size_t i = 0; i < y.size(); i++) {
            double diff = fabs(y_corr[i] - y_pred[i]);
            double scale = max(fabs(y[i]), abs_tol);
            err = max(err, diff / scale);
        }

        if (err <= rel_tol) {
            // Accept step
            t += h;
            y = y_corr;
            result.time.push_back(t);
            result.values.push_back(y);

            if (tspan_index < tspan.size() && t == tspan[tspan_index]) {
                tspan_index++;
            }
        }

        // Adjust step size
        if (err == 0) {
            h = min(h * 2.0, hmax);
        } else {
            h = max(min(h * 0.9 * sqrt(rel_tol / err), hmax), hmin);
        }

        if (h < hmin) {
            cerr << "Warning: Step size below minimum. Exiting loop." << endl;
            break;
        }
    }

    return result;
}


