#include "MPCI.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <cstring>
//#include "../AhmedMAtallah/Chebyshev-Picard-Method/include/SHM.h"
//#include "../AhmedMAtallah/Chebyshev-Picard-Method/include/MCPIIOM.h"
//#include "../AhmedMAtallah/Chebyshev-Picard-Method/include/Basic.h"

constexpr int Nstates = 6;
//constexpr int MaxIter = 20;
//constexpr double mcpi_tol = 1.0e-17;

std::vector<std::vector<double>> MPCI( // We are assuming the t_span already has the gauss lobato nodes. So we just need to know the segree to know where the segments are
    std::vector<double> (*func)(double, const std::vector<double>&, double, double, double),
    const std::vector<double>& t_span,
    const std::vector<double>& y0,
    double mcpi_tol = 1e-9,
    double A=4.0,
    double m=2.4,
    double C_D=2.2,
    int N=16,
    int MaxIter 
) {

    //TODO maybe it would be easier to just look at the Matlab Implementation and try to implement that in cpp myself...

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
}