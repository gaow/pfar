// Gao Wang and Kushal K. Dey (c) 2016
#include "pfa.hpp"
#include <stdexcept>

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
// @param niter [int_pt] number of iterations
// @param loglik [double_pt] log likelihood (return)
// @param L [N, K] Loading matrix (return)

extern "C" int pfa_em(double *, double *, double *, double *, double *,
                      int *, int *, int *, int *, double *, int *,
                      double *, double *);

int pfa_em(double * X, double * F, double * P, double * q, double * omega,
           int * N, int * J, int * K, int * C, double * tol, int * niter,
           double * loglik, double * L) {
  *niter = 0;
  double prev = 0, curr = 0;
  PFA model(X, F, P, q, omega, *N, *J, *K, *C);
  // model.print();
  while (true) {
    model.get_log_delta_given_nkq();
    curr = model.get_loglik_prop();
    if (curr < prev && *niter > 0) {
      throw std::runtime_error("EM algorithm error: likelihood decrease!");
    }
    if (curr - prev < *tol && *niter > 0)
      break;
    prev = curr;
    model.update_weights();
    model.update_F();
    *niter++;
  }
  return 0;
}
