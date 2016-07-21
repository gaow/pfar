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

//! Vectorize pairwise factor frequency matrix.
//! The input matrix has upper.tri portion all zero
//! which will be removed in the conversion
inline arma::vec kk2_to_kk1(arma::mat kk2) {
  arma::vec kk1 = arma::nonzeros(arma::vectorise(kk2));
  return kk1;
}

class PFA {
public:
  PFA(double * X, double * F, double * P, double * q,
      int N, int J, int K, int C):
    D(X, N, J, false, true), F(F, K, J, false, true),
    P(P, K, K, false, true), q(q, C, false, true) {
    s = arma::vectorise(arma::stddev(D));
  }
    // mat(aux_mem*, n_rows, n_cols, copy_aux_mem = true, strict = true)
  ~PFA() {}

  void print() {
    D.print("Data Matrix:");
    F.print("Factor Matrix:");
    P.print("Factor frequency Matrix:");
    q.print("Membership grids:");
    s.print("Estimated standard deviation of data columns (features):");
  }

  void get_delta_given_nkq() {
    // this computes delta up to a normalizing constant
    // this results in a k1k2 by q by N tensor
  }

  void update_pik() {

  }

  void update_omegaq() {

  }

  void update_F() {

  }

  void get_loglik() {

  }

private:
  arma::mat D;
  arma::mat F;
  arma::mat P;
  arma::vec q;
  arma::vec s;
};
#endif
