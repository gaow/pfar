// Gao Wang and Kushal K. Dey (c) 2016
#include "pfa.hpp"

//! EM algorithm for paired factor analysis
// @param X [N, J] observed data matrix
// @param F [K, J] initial factor matrix
// @param P [K, K] initial frequency matrix of factor pairs, an upper diagonal matrix
// @param q [float, 1] initial vector of possible membership loadings, a discrete set
// @param N [int_pt] number of rows of matrix X
// @param J [int_pt] number of columns of matrix X and F
// @param K [int_pt] number of rows of matrix F and P
// @param C [int_pt] number of elements in q

extern "C" int pfa_em(double *, double *, double *, double *,
                      int *, int *, int *, int *);

int pfa_em(double * X, double * F, double * P, double * q,
           int * N, int * J, int * K, int * C) {
  PFA model(X, F, P, q, *N, *J, *K, *C);
  model.print();
  return 0;
}
