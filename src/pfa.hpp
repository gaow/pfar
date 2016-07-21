#ifndef _PFA_HPP
#define _PFA_HPP

#include <armadillo>
#include <iostream>

static const double INV_SQRT_2PI = 0.3989422804014327;
static const double INV_SQRT_2PI_LOG = -0.91893853320467267;

inline double normal_pdf(double x, double m, double s)
{
  double a = (x - m) / s;
  return INV_SQRT_2PI / s * std::exp(-0.5 * a * a);
};

inline double normal_pdf_log(double x, double m, double s)
{
  double a = (x - m) / s;
  return INV_SQRT_2PI_LOG - std::log(s) -0.5 * a * a;
};

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
