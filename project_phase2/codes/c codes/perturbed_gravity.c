/*
*  AUTHORS:          Robyn Woollands (robyn.woollands@gmail.com)
*  DATE WRITTEN:     June 2016
*  LAST MODIFIED:    June 2016
*  AFFILIATION:      Department of Aerospace Engineering, Texas A&M University, College Station, TX
*  DESCRIPTION:      Computes gravity using the variable fidelity force approximations
*
* INPUT:
*    t     -- Time (s)
*    Xo    -- State (position and velocity)
*    err   -- Picard iteration relative error
*    i     -- index along array for current segment
*    M     -- Number of sample points
*    deg   -- Gravity degree
*    hot   -- Hot start switch condition
*    tol   -- Tolerance
*    itr   -- Picard iteration counter
*    Feval -- Function evaluation counter
*
* OUTPUTS:
*    G   -- Gravitational acceleration
*
* REFERENCES:
* 1. Macomber, B., Probe, A., Woollands, R., Read, J., and Junkins, J., "Enhancements of
*    Modified Chebyshev Picard Iteration for Perturbed Orbit Propagation", Computational
*    Modelling in Engineering & Sciences, Vol. 111, pp, 29-64, 2016.
*/

#include "perturbed_gravity.h"
#include "EGM2008.h"
#include "const.h"
#include "c_functions.h"
#include "matrix_loader.h"
#include "radial_gravity.h"

#define debug_grav 0
#define debug_grav_itr 0

 void perturbed_gravity(double t, double* Xo, double err, int i, int M, double deg, int hot, double* G, double tol, int* itr, double* Feval) {
    double Gapprox[3] = {0.0};
    static double del_G[3 * (Nmax + 1)];
    static int MODEL;

    // Initialization
    if (*itr == 0 && hot == 0) {
        MODEL = 0;
    } else if (*itr == 0 && hot == 1) {
        MODEL = 0;
    }

    // Use only J2 term
    if (debug_grav == 1 && i == 1) {
        printf("Using J2 only\n");
    }
    Grav_Approx(t, Xo, G, Feval);

    if (i == M + 1) {
        *itr = *itr + 1;
    }
    return;
}


 void Grav_Approx(double t, double* X, double* dX, double* Feval) {
    double r = sqrt(pow(X[0], 2) + pow(X[1], 2) + pow(X[2], 2));
    double J2 = 1082.63e-6;

    double x_r_1 = X[0] / r;
    double y_r_1 = X[1] / r;
    double z_r_1 = X[2] / r;
    double z_r_2 = pow((X[2] / r), 2);

    double aTB[3] = {0.0};
    double aJ2[3] = {0.0};

    for (int i = 0; i <= 2; i++) {
        aTB[i] = -(C_MU / pow(r, 3)) * X[i];
    }

    aJ2[0] = -3.0 / 2.0 * J2 * (C_MU / pow(r, 2)) * pow((C_Req / r), 2) * (1.0 - 5.0 * z_r_2) * x_r_1;
    aJ2[1] = -3.0 / 2.0 * J2 * (C_MU / pow(r, 2)) * pow((C_Req / r), 2) * (1.0 - 5.0 * z_r_2) * y_r_1;
    aJ2[2] = -3.0 / 2.0 * J2 * (C_MU / pow(r, 2)) * pow((C_Req / r), 2) * (3.0 - 5.0 * z_r_2) * z_r_1;

    for (int i = 0; i <= 2; i++) {
        dX[i] = aTB[i] + aJ2[i];
    }

    Feval[1] = Feval[1] + 1.0;
    return;
}


 void Grav_Full(double t, double* Xo, double* acc, double tol, double deg, double* Feval){

   double state[6]  = {0.0};
   double dstate[6] = {0.0};

   for (int jj=0; jj<=5; jj++){
     state[jj] = Xo[jj];
     if (jj>2){
       state[jj] = 0.0;
     }
   }

   double grav = 0.0;
   radial_gravity(Xo,tol,deg,&grav);
   EGM2008(state, &dstate[3], grav);
   Feval[0] = Feval[0] + pow(grav,2)/pow(deg,2);

   for (int jj=0; jj<=2; jj++){
     acc[jj] = dstate[jj+3];
   }

   return;
 }
