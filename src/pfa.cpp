// Gao Wang and Kushal K. Dey (c) 2016
#include "pfa.hpp"

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

extern "C" int pfa_em(double *, double *, double *, double *, double *,
                      int *, int *, int *, int *);

int pfa_em(double * X, double * F, double * P, double * q, double * omega,
           int * N, int * J, int * K, int * C) {
  PFA model(X, F, P, q, omega, *N, *J, *K, *C);
  model.get_log_delta_given_nkq();
  model.update_pik_omegaq();
  model.print();
  return 0;
}
