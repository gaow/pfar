// Gao Wang and Kushal K. Dey (c) 2016

#include <pfa.hpp>

//! EM algorithm for paired factor analysis
// @param X [int, int] observed data matrix
// @param X_size1 [int_pt] number of rows of matrix X
// @param X_size2 [int_pt] number of columns of matrix X
extern "C" int pfa_em(double *, int*, int*);

int pfa_em(double * X, int* X_size1, int* X_size2) {
  int N = *X_size1, K = *X_size2;
  arma::mat A(X, N, K, false, true);
  A.print();
  return 0;
}
