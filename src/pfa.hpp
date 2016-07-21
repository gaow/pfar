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
  PFA(double * X, double * F, double * P, double * q, double * omega,
      int N, int J, int K, int C):
    // mat(aux_mem*, n_rows, n_cols, copy_aux_mem = true, strict = true)
    D(X, N, J, false, true), F(F, K, J, false, true),
    P(P, K, K, false, true), q(q, C, false, true),
    omega(omega, C, false, true)
  {
    s = arma::vectorise(arma::stddev(D));
  }
  ~PFA() {}

  void print() {
    D.print("Data Matrix:");
    F.print("Factor Matrix:");
    P.print("Factor frequency Matrix:");
    q.print("Membership grids:");
    omega.print("Membership grid weights:");
    s.print("Estimated standard deviation of data columns (features):");
    log_delta.print("Current log(delta) tensor:");
    avg_delta.print("Current delta averaged over samples:");
  }

  void get_log_delta_given_nkq() {
    // this computes delta up to a normalizing constant
    // this results in a N by k1k2 by q tensor of loglik
    log_delta.reshape(D.n_rows, int((1 + P.n_rows) * P.n_rows / 2), q.n_elem);
    for (size_t qq = 0; qq < q.n_elem; qq++) {
      size_t col_cnt = 0;
      for (size_t k1 = 0; k1 < P.n_rows; k1++) {
        for (size_t k2 = 0; k2 <= k1; k2++) {
          // given k1, k2 and q, density is a N-vector
          // in the end of the loop, the vector should populate a column of a slice of the tensor
          arma::vec Dn_llik = arma::zeros<arma::vec>(D.n_rows);
          for (size_t j = 0; j < D.n_cols; j++) {
            arma::vec density = D.col(j);
            double m = q.at(qq) * F.at(k1, j) + (1 - q.at(qq)) * F.at(k2, j);
            double sig = s.at(j);
            Dn_llik += density.transform( [=](double x) { return (normal_pdf_log(x, m, sig)); } );
          }
          Dn_llik = Dn_llik + (std::log(P.at(k1, k2)) + std::log(omega.at(qq)));
          // populate the tensor
          log_delta.slice(qq).col(col_cnt) = Dn_llik;
          col_cnt++;
        }
      }
    }
    // this results in a K1K2 by q matrix of avglik
    // which equals pi_k1k2 * omega_q
    avg_delta.reshape(int((1 + P.n_rows) * P.n_rows / 2), q.n_elem);
    arma::cube slice_rowsums = arma::sum(arma::exp(log_delta));
    for (size_t qq = 0; qq < q.n_elem; qq++) {
      avg_delta.col(qq) = arma::vectorise(slice_rowsums.slice(qq)) / D.n_rows;
    }
  }

  void update_pik_omegaq() {
    // sum over q grids
    arma::vec pik1k2 = arma::sum(avg_delta, 1);
    pik1k2 = pik1k2 / arma::sum(pik1k2);
    size_t col_cnt = 0;
    for (size_t k1 = 0; k1 < P.n_rows; k1++) {
      for (size_t k2 = 0; k2 <= k1; k2++) {
        P.at(k1, k2) = pik1k2.at(col_cnt);
        col_cnt++;
      }
    }
    // sum over k1, k2
    omega = arma::vectorise(arma::sum(avg_delta));
    omega = omega / arma::sum(omega);
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
  arma::vec omega;
  arma::vec s;
  // N by K1K2 by q tensor
  arma::cube log_delta;
  // K1K2 by q matrix
  arma::mat avg_delta;
};
#endif
