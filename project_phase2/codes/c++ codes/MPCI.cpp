#include "MPCI.h"
#include <Eigen/Dense>
#include <iostream>
// #include "../AhmedMAtallah/Chebyshev-Picard-Method/include/SHM.h"
// #include "../AhmedMAtallah/Chebyshev-Picard-Method/include/MCPIIOM.h"
// #include "../AhmedMAtallah/Chebyshev-Picard-Method/include/Basic.h"

constexpr int Nstates = 6;
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// constexpr int MaxIter = 20;
// constexpr double mcpi_tol = 1.0e-17;

std::vector<std::vector<double>> MPCI(
    std::vector<double> (*odefun)(double, const std::vector<double> &, double, double, double),
    const std::vector<double> &tspan,
    const std::vector<double> &x0,
    double abs_tol,
    double rel_tol,
    double Sec, // in satellites Sec is the Orbital period/num_segments. -1 means use tspan.back()
    int MaxIter,
    int N,
    double Asat,
    double m,
    double C_D)
{

    N = N - 1; // Adjust for 0-based indexing
    double tstart = 0.0;
    double tend = (Sec == -1) ? tspan.back() : Sec;

    int output_length = N * std::ceil(tspan.back() / tend) + 1;
    std::vector<std::vector<double>> xout(output_length, std::vector<double>(6));
    std::vector<double> tout(output_length);
    xout[0] = x0; // Set initial condition

    int firstpos = 0;
    int lastpos;

    Eigen::RowVectorXd tau = (Eigen::RowVectorXd::LinSpaced(N + 1, N, 0).array() * M_PI / N).cos();

    // Map x0 directly as a row vector
    Eigen::RowVectorXd x0_row = Eigen::Map<const Eigen::RowVectorXd>(x0.data(), x0.size());

    // Create xtinit matrix: each row is a copy of x0
    Eigen::MatrixXd xtinit(tau.cols(), x0.size());
    xtinit.rowwise() = x0_row;

    // Create expanded X0 matrix
    Eigen::MatrixXd X0_expanded = Eigen::MatrixXd::Zero(N + 1, x0.size());
    X0_expanded.row(0) = Eigen::Map<const Eigen::RowVectorXd>(x0.data(), x0.size());

    bool lastrunflag = false;

    while (true)
    {
        Eigen::MatrixXd xold = xtinit; // Initial guess for this segment
        Eigen::MatrixXd xnew;
        Eigen::MatrixXd bi;
        double om2 = (tend - tstart) / 2.0;
        double om1 = (tend + tstart) / 2.0;

        // Create W matrix (identity with 0.5 on corners)
        Eigen::MatrixXd W = Eigen::MatrixXd::Identity(tau.size(), tau.size());
        W(0, 0) = 0.5;
        W(tau.size() - 1, tau.size() - 1) = 0.5;

        // Creat T matrix
        Eigen::VectorXd k_indices = Eigen::VectorXd::LinSpaced(N, 0, N - 1);
        Eigen::RowVectorXd acos_tau = tau.array().acos();
        Eigen::MatrixXd T = (k_indices * acos_tau).array().cos().transpose();

        // Create Tm1: Chebyshev polynomials at acos(-1) = π
        Eigen::RowVectorXd Tm1 = (Eigen::RowVectorXd::LinSpaced(N + 1, 0, N).array() * M_PI).cos();

        // Create L matrix: first row is Tm1, rest are zeros
        Eigen::MatrixXd L = Eigen::MatrixXd::Zero(N + 1, N + 1);
        L.row(0) = Tm1;

        // Create s_ vector: s_ = 1./(4:2:2*N)
        Eigen::VectorXd s_ = (Eigen::VectorXd::LinSpaced(N - 1, 4, 2 * N)).cwiseInverse();

        // Create S_3 matrix with upper diagonal
        Eigen::MatrixXd S_3 = Eigen::MatrixXd::Zero(N + 1, N + 1);
        Eigen::VectorXd upper_diag(N);
        upper_diag[0] = 0.0;
        upper_diag[1] = -0.5;
        upper_diag.tail(N - 2) = -s_.head(N - 2); // Remaining elements: -s_[0] through -s_[12]
        for (int i = 1; i < N; ++i)
        {
            S_3(i, i + 1) = upper_diag[i];
        }
        // std::cout << "Upper diagonal of S_3:\n" << upper_diag.transpose() << std::endl;

        // Create S_2 matrix with lower diagonal
        Eigen::MatrixXd S_2 = Eigen::MatrixXd::Zero(N + 1, N + 1);
        Eigen::VectorXd lower_diag(N);
        lower_diag[0] = 1.0;
        lower_diag.tail(N - 1) = s_;
        for (int i = 1; i < N + 1; ++i)
        {
            S_2(i, i - 1) = lower_diag[i - 1];
        }
        // std::cout << "Lower diagonal of S_2:\n" << lower_diag.transpose() << std::endl;
        //  Combine S_2 and S_3
        Eigen::MatrixXd S_1 = S_2 + S_3;

        // Extract S matrix: S = S_1(:,1:N) and set first row
        Eigen::MatrixXd S = S_1.leftCols(N);
        S.row(0) = Eigen::RowVectorXd::Zero(N);
        S(0, 0) = 0.25; // S(1,:) = [1/4, zeros(1,N-1)]

        // Create A matrix: A = (T'*W*T)\T'*W
        Eigen::MatrixXd A = (T.transpose() * W * T).ldlt().solve(T.transpose() * W);

        // Reallocate T matrix for the new dimensions (0:N instead of 0:N-1)
        k_indices = Eigen::VectorXd::LinSpaced(N + 1, 0, N);
        T = (k_indices * acos_tau).array().cos().transpose();

        double scaled_error;
        // double eAbs = 1e15;
        // double eRel = 1e15;
        int i = 0;
        const double CONVERGENCE_SAFETY = 0.1;  // 100× more stringent (or use 1.0 for no safety)

        //while ((eAbs > abs_tol || eRel > rel_tol) && i < MaxIter)
        while (i < MaxIter)
        {
            // Compute F matrix: evaluate odefun at each transformed time
            Eigen::MatrixXd F(tau.cols(), 6); // (N+1) × 6 matrix

            for (int j = 0; j < tau.cols(); ++j)
            {
                double t_j = om2 * tau[j] + om1; // Transform tau to actual time

                // Extract state vector for this node from xold
                std::vector<double> state_j(6);
                for (int k = 0; k < 6; ++k)
                {
                    state_j[k] = xold(j, k);
                }

                // Evaluate ODE function
                std::vector<double> f_result = odefun(t_j, state_j, Asat, m, C_D);

                // Store result in F matrix
                for (int k = 0; k < 6; ++k)
                {
                    F(j, k) = f_result[k];
                }
            }

            // Compute P1 = om2*(eye(N+1) - L) * S
            Eigen::MatrixXd P1 = om2 * (Eigen::MatrixXd::Identity(N + 1, N + 1) - L) * S;

            // Compute bi = X0 + P1 * A * F
            bi = X0_expanded + P1 * A * F;

            // Compute xnew = T * bi
            xnew = T * bi;

            // Compute convergence criteria
            // eAbs = (xnew - xold).cwiseAbs().maxCoeff();
            // eRel = ((xnew - xold).cwiseAbs().cwiseQuotient(xold.cwiseAbs().cwiseMax(xnew.cwiseAbs()))).maxCoeff();

            double scaled_error = ((xnew - xold).cwiseAbs().array() / (abs_tol + rel_tol * xold.cwiseAbs().cwiseMax(xnew.cwiseAbs()).array())).maxCoeff();

            if (scaled_error <= CONVERGENCE_SAFETY) {
                    break;  // Converged
                }
            // eAbs = (xnew - xold).cwiseAbs().maxCoeff();
            // Eigen::MatrixXd denominator = xold.cwiseAbs().cwiseMax(xnew.cwiseAbs()).cwiseMax(1e-15);
            // eRel = ((xnew - xold).cwiseAbs().cwiseQuotient(denominator)).maxCoeff();

            // if (i % 10 == 0 || i < 5)
            // {
            //     std::cout << "Iter " << i << ": eAbs=" << eAbs << " (target: " << abs_tol << ")" << std::endl;
            //     std::cout << "         eRel=" << eRel << " (target: " << rel_tol << ")" << std::endl;

            //     if (i < 3)
            //     {
            //         // Print some sample values for first few iterations
            //         std::cout << "Sample xnew: [" << xnew(0, 0) << ", " << xnew(0, 1) << ", " << xnew(0, 2) << "]" << std::endl;
            //         std::cout << "Sample xold: [" << xold(0, 0) << ", " << xold(0, 1) << ", " << xold(0, 2) << "]" << std::endl;
            //     }
            // }
            // Update xold for next iteration
            xold = xnew;

            i++; // Increment iteration counter
        }

        // Check if we failed to converge
        if (i >= MaxIter)
        {
            std::cout << "MPCI failed to converge after " << MaxIter << " iterations" << std::endl;
            return std::vector<std::vector<double>>();
        }

        // TODO: Handle Case 1 - if tspan is a vector of specific time points
        // Need to transform tspan to tau coordinates and evaluate T*bi at those points

        // Case 2: tspan is just [start, end] - use Gauss-Lobatto nodes
        lastpos = firstpos + N;

        // Copy xnew (skipping first row) into xout using std::copy
        for (int i = 1; i < xnew.rows(); ++i)
        {
            int xout_idx = firstpos + i;
            // Copy entire row at once
            for (int j = 0; j < 6; ++j)
            {
                xout[xout_idx][j] = xnew(i, j);
            }
        }

        // Update for next segment
        firstpos = lastpos;
        tstart = tend;
        tend = tstart + Sec;

        // Check if this was the last run
        if (lastrunflag)
        {
            break; // Exit the while(true) loop
        }

        // Check if we've reached the final segment
        if (tend > tspan.back())
        {
            tend = tspan.back();
            lastrunflag = true;
        }

        // Update initial conditions from the last state
        std::vector<double> new_x0(6);
        for (int j = 0; j < 6; ++j)
        {
            new_x0[j] = xnew(xnew.rows() - 1, j); // Last row of xnew
        }

        // Update xtinit with new initial condition
        xtinit.rowwise() = Eigen::Map<const Eigen::RowVectorXd>(new_x0.data(), 6);

        // Update X0_expanded
        X0_expanded.row(0) = Eigen::Map<const Eigen::RowVectorXd>(new_x0.data(), 6);
        X0_expanded.bottomRows(N).setZero();
    }

    // Trim xout to actual used size
    // Validate that we used the expected amount of space
    if (firstpos != output_length - 1)
    {
        std::cout << "Warning: Expected output length " << output_length
                  << " but used " << firstpos + 1 << " elements" << std::endl;
    }

    return xout;
}
// TODO maybe it would be easier to just look at the Matlab Implementation and try to implement that in cpp myself...
/*
    int N = n_points - 1; // Degree
    int M = N + 1;
    std::vector<std::vector<double>> result;

    // Initial condition
    double x0[Nstates] = {0.0};
    for (int kk = 0; kk < Nstates; kk++)
        x0[kk] = y0[kk];

    double Xo[M * Nstates] = {0.0};
    for (int m = 0; m < M; m++)
        for (int kk = 0; kk < Nstates; kk++)
            Xo[IDX2F(m + 1, kk + 1, M)] = y0[kk];

    // Build TAU vector
    double TAU[M] = {0.0};
    for (int i = 0; i < M; i++)
        TAU[i] = cos((double)i * Pi / (double)(M - 1) + Pi);

    double xAdd[M * Nstates] = {0.0};
    double G[M * Nstates] = {0.0};
    double Xn[M * Nstates] = {0.0};

    // MCPI coefficients
    double Im[M * M];
    MCPI_CoeffsI(N, M, Im);
    double ImT[M * M];
    trans(Im, ImT, M, M, M);

    // Time arrays
    // TODO here's where you need to figure it out. You need to switch between what he's doing with the arrays.
    // He called tspan the tmax. For us that's t_span.back. He generates a vector of all the gaus lobato nodes at the start of the loop with the a,b parameters.
    // We need to simply take those points from the t_span vector. Also, we need to replace accGravity with the func that we got.
    int num_segments = t_span.size()/M;
    double timeSubArr[num_segments] = {0.0};
    double timeAddArr[num_segments] = {0.0};
    double timearray[num_segments]={0.0};
    double Tsum = {0.0};
    double timeArray[M*num_segments] = {0};
    double timeArraySeg[M]={0};

    for (int IT = 0; IT < num_segments; IT++) {
        double b = ((double)IT + 1.0) * t_span.back() / num_segments;
        double a = (double)IT * t_span.back() / num_segments;
        double timeSub = (b - a) / 2.0;
        double timeAdd = (b + a) / 2.0;
        timeSubArr[IT] = timeSub;
        timeAddArr[IT] = timeAdd;

        double timeArraySeg[M] = {0.0};
        for (int k = 0; k < M; k++)
            timeArraySeg[k] = timeSubArr[IT] * TAU[k] + timeAddArr[IT];

        int loopCount = 0;
        double temp = 1.0;

        while (loopCount < MaxIter) {
            accGravity(M, N, timeArraySeg, Xo, G);
            matmul(ImT, G, xAdd, M, M, Nstates);
            errorAndUpdate(M, timeSub, Nstates, x0, Xo, Xn, xAdd, temp);

            if (loopCount >= MaxIter)
                break;
            loopCount++;
        }

        // Update initial conditions for next segment
        for (int kk = 0; kk < Nstates; kk++)
            x0[kk] = Xn[IDX2F(M, kk + 1, M)];
        for (int m = 0; m < M; m++)
            for (int kk = 0; kk < Nstates; kk++)
                Xo[IDX2F(m + 1, kk + 1, M)] = x0[kk];

        // Store results for this segment
        for (int k = 0; k < M; k++) {
            std::vector<double> state(Nstates);
            for (int kk = 0; kk < Nstates; kk++)
                state[kk] = Xn[IDX2F(k + 1, kk + 1, M)];
            result.push_back(state);
        }
    }
    return result;
}*/