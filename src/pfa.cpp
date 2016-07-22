// Gao Wang and Kushal K. Dey (c) 2016
#include "pfa.hpp"
#include <iostream>

//! EM algorithm for paired factor analysis
// @param X [N, J] observed data matrix
// @param F [K, J] initial factor matrix
// @param P [K, K] initial frequency matrix of factor pairs, an upper diagonal matrix
// @param q [C, 1] initial vector of possible membership loadings, a discrete set
// @param omega [C, 1] initial weight of membership loadings, a discrete set corresponding to q
// @param N [int_pt] number of rows of matrix X
// @param J [int_pt] number of columns of matrix X and F
// @param K [int_pt] number of rows of matrix F and P
// @param C [int_pt] number of elements in q
// @param tol [double_pt] tolerance for convergence
// @param maxiter [int_pt] maximum number of iterations
// @param niter [int_pt] number of iterations
// @param track_c [maxiter, 1] track of convergence (return)
// @param L [N, K] Loading matrix (return)
// @param status [int_pt] return status, 0 for good, 1 for error (return)

extern "C" int pfa_em(double *, double *, double *, double *, double *,
                      int *, int *, int *, int *, double *, int *, int *,
                      double *, double *, int *);

int pfa_em(double * X, double * F, double * P, double * q, double * omega,
           int * N, int * J, int * K, int * C, double * tol, int * maxiter, int * niter,
           double * track_c, double * L, int * status) {
  *niter = 0;
  PFA model(X, F, P, q, omega, *N, *J, *K, *C);
  // model.print();
  while (*niter <= *maxiter) {
    model.get_log_delta_given_nkq();
    track_c[*niter] = model.get_loglik_prop();
    if (*niter > 0) {
      // check monotonicity
      if (track_c[*niter] < track_c[(*niter - 1)]) {
        std::cerr << "[ERROR] likelihood decreased in EM algorithm!" << std::endl;
        *status = 1;
        break;
      }
      // converged
      if (track_c[*niter] - track_c[(*niter - 1)] < *tol)
        break;
    }
    if (*niter == *maxiter) {
      // did not converge
      *status = 1;
      break;
    }
    model.update_weights();
    model.update_F();
    (*niter)++;
  }
  if (*status) std::cerr << "[WARNING] EM algorithm failed to converge!" << std::endl;
  return 0;
}
