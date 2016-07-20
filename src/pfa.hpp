#ifndef _PFA_HPP
#define _PFA_HPP

#include <armadillo>
#include <iostream>

class PFA {
public:
  PFA(double * X, int N, int K):
    D(X, N, K, false, true) {}
    // mat(aux_mem*, n_rows, n_cols, copy_aux_mem = true, strict = true)
  ~PFA() {}

  void Print() {
    D.print("Data Matrix:");
  }

private:
  arma::mat D;
};
#endif
