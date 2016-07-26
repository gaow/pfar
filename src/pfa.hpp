#ifndef _PFA_HPP
#define _PFA_HPP

#include <armadillo>
#include <map>
#include <omp.h>
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
  PFA(double * cX, double * cF, double * cP, double * cQ, double * omega,
      double * cLout, int N, int J, int K, int C):
    // mat(aux_mem*, n_rows, n_cols, copy_aux_mem = true, strict = true)
    D(cX, N, J, false, true), F(cF, K, J, false, true),
    P(cP, K, K, false, true), q(cQ, C, false, true),
    omega(omega, C, false, true), L(cLout, N, K, false, true)
  {
    s = arma::vectorise(arma::stddev(D));
    L.set_size(D.n_rows, F.n_rows);
    W.set_size(F.n_rows, F.n_rows);
    delta.set_size(int((F.n_rows - 1) * F.n_rows / 2), q.n_elem, D.n_rows);
    pi_mat.set_size(int((P.n_rows - 1) * P.n_rows / 2), q.n_elem);
    n_threads = 1;
    n_updates = std::make_pair(0, 0);
    // set factor pair coordinates in the tensor
    // to avoid having to compute it at each iteration
    for (size_t k1 = 0; k1 < F.n_rows; k1++) {
      for (size_t k2 = 0; k2 < k1; k2++) {
        // (b - 1) * b / 2 - ((b - 1) - a)
        size_t k1k2 = size_t(k1 * (k1 - 1) / 2 + k2);
        F_pair_coord[std::make_pair(k1, k2)] = k1k2;
        F_pair_coord[std::make_pair(k2, k1)] = k1k2;
      }
    }
  }
  ~PFA() {}

  void print(std::ostream& out, int info) {
    if (info == 0) {
      D.print(out, "Data Matrix:");
      q.print(out, "Membership grids:");
      s.print(out, "Estimated standard deviation of data columns (features):");
    }
    if (info == 1) {
      F.print(out, "Factor Matrix:");
      P.print(out, "Factor frequency Matrix:");
      L.print(out, "Loading matrix:");
      W.print(out, "E[L'L] matrix:");
      omega.print(out, "Membership grid weights:");
    }
    if (info == 2) {
      pi_mat.print(out, "delta averaged over samples:");
    }
  }

  void set_threads(int n) {
    n_threads = n;
  }

  void get_loglik_given_nkq() {
    // this computes loglik and delta
    // this results in a N by k1k2 by q tensor of loglik
    // and a k1k2 by q by N tensor of delta
    // and a k1k2 by q matrix of pi_mat
    arma::cube loglik_mat;
    loglik_mat.set_size(int((F.n_rows - 1) * F.n_rows / 2), q.n_elem, D.n_rows);
#pragma omp parallel for num_threads(n_threads)
    for (size_t qq = 0; qq < q.n_elem; qq++) {
      for (size_t k1 = 0; k1 < F.n_rows; k1++) {
        for (size_t k2 = 0; k2 < k1; k2++) {
          // given k1, k2 and q, density is a N-vector
          // in the end of the loop, the vector should populate a column of a slice of the tensor
          arma::vec Dn_llik = arma::zeros<arma::vec>(D.n_rows);
          arma::vec Dn_delta = arma::zeros<arma::vec>(D.n_rows);
          for (size_t j = 0; j < D.n_cols; j++) {
            arma::vec density = D.col(j);
            double m = q.at(qq) * F.at(k2, j) + (1 - q.at(qq)) * F.at(k1, j);
            double sig = s.at(j);
            Dn_llik += density.transform( [=](double x) { return (normal_pdf_log(x, m, sig)); } );
          }
          size_t k1k2 = size_t(k1 * (k1 - 1) / 2 + k2);
          // populate the loglik and the delta tensor
          if (n_updates == std::make_pair(0, 0)) {
            // No update on pi_mat is available yet: approximate it under independence assumption
            Dn_delta = arma::exp(Dn_llik) * (P.at(k1, k2) * omega.at(qq));
          }
          else {
            Dn_delta = arma::exp(Dn_llik) * pi_mat.at(k1k2, qq);
          }
          // FIXME: this is slow, due to the cube/slice structure
          for (size_t n = 0; n < D.n_rows; n++) {
            loglik_mat.slice(n).at(k1k2, qq) = Dn_llik.at(n);
            delta.slice(n).at(k1k2, qq) = Dn_delta.at(n);
          }
        }
      }
    }
    // Compute log likelihood and pi_mat
    arma::vec loglik_vec;
    loglik_vec.set_size(D.n_rows);
#pragma omp parallel for num_threads(n_threads)
    for (size_t n = 0; n < D.n_rows; n++) {
      delta.slice(n) = delta.slice(n) / arma::accu(delta.slice(n));
      loglik_vec.at(n) = std::log(arma::accu(arma::exp(loglik_mat.slice(n))));
    }
    pi_mat = arma::sum(delta, 2) / D.n_rows;
    loglik = arma::accu(loglik_vec);
  }

  double get_loglik() {
    // current log likelihood
    return loglik;
  }

  void update_weights() {
    // update P and omega
    // sum over q grids
    arma::vec pik1k2 = arma::sum(pi_mat, 1);
    pik1k2 = pik1k2 / arma::sum(pik1k2);
#pragma omp parallel for num_threads(n_threads)
    for (size_t k1 = 0; k1 < P.n_rows; k1++) {
      for (size_t k2 = 0; k2 < k1; k2++) {
        size_t k1k2 = size_t(k1 * (k1 - 1) / 2 + k2);
        P.at(k1, k2) = pik1k2.at(k1k2);
      }
    }
    // sum over (k1, k2)
    omega = arma::vectorise(arma::sum(pi_mat));
    omega = omega / arma::sum(omega);
    // keep track of iterations
    n_updates.first += 1;
  }

  void update_LF() {
    // F, the K by J matrix, is to be updated here
    // L, the N by K matrix, is also to be updated
    // Need to compute 2 matrices in order to solve F
    // The loading, L is N X K matrix; W = L'L is K X K matrix
    L.fill(0);
    for (size_t k = 0; k < F.n_rows; k++) {
      // I. First we compute the k-th column for E(L), the N X K matrix:
      // generate the proper input for 1 X K %*% K X N
      // where the 1 X K matrix is loadings Lk consisting of q or (1-q)
      // and the K X N matrix is the delta for a corresponding subset of a slice from delta
      // II. Then we compute the diagonal elements for E(W), the K X K matrix
      // Because we need to sum over all N and we have computed this before,
      // we can work with pi_mat (K1K2 X q) instead of delta the tensor
      // we need to loop over the q slices
#pragma omp parallel for num_threads(n_threads)
      for (size_t qq = 0; qq < q.n_elem; qq++) {
        // I. ........................................
        // create the left hand side Lk1, a 1 X K matrix
        double qi = q.at(qq);
        arma::mat Lk1(1, F.n_rows, arma::fill::zeros);
        for (size_t i = 0; i < F.n_rows; i++) {
          if (i < k) Lk1.at(0, i) = 1.0 - qi;
          if (i > k) Lk1.at(0, i) = qi;
        }
        // create the right hand side Lk2, a K X N matrix
        // from a slice of the tensor delta
        // where the rows are N's, the columns corresponds to
        // k1k2, k1k3, k2k3, k1k4, k2k4, k3k4, k1k5 ... along the lower triangle matrix P
        // use F_pair_coord to get proper index for data from the tensor
        arma::mat Lk2(F.n_rows, D.n_rows, arma::fill::zeros);
        for (size_t i = 0; i < F.n_rows; i++) {
          for (size_t n = 0; n < D.n_rows; n++) {
            if (k != i)
              Lk2.at(i, n) = delta.slice(n).at(F_pair_coord[std::make_pair(k, i)], qq);
          }
        }
        // Update the k-th column of L
        L.col(k) += arma::vectorise(Lk1 * Lk2);
        // II. ........................................
        for (size_t i = 0; i < F.n_rows; i++) {
          if (i < k) Lk1.at(0, i) = (1.0 - qi) * (1.0 - qi);
          if (i > k) Lk1.at(0, i) = qi * qi;
        }
        arma::mat Lk3(F.n_rows, 1, arma::fill::zeros);
        for (size_t i = 0; i < F.n_rows; i++) {
          if (k != i) Lk3.at(i, 0) = D.n_rows * pi_mat.at(F_pair_coord[std::make_pair(k, i)], qq);
        }
        // Update E(W_kk)
        W.at(k, k) += arma::as_scalar(Lk1 * Lk3);
      }
    }
    // III. Now we compute off-diagonal elements for E(W), the K X K matrix
    // it involves on the LHS a vector of [q1(1-q1), q2(1-q2) ...]
    // and on the RHS for each pair of (k1, k2) the corresponding row from pi_mat
    arma::vec LHS = q.transform( [](double val) { return (val * (1.0 - val)); } );
#pragma omp parallel for num_threads(n_threads)
    for (size_t k1 = 0; k1 < F.n_rows; k1++) {
      for (size_t k2 = 0; k2 < k1; k2++) {
        arma::vec RHS = arma::vectorise(D.n_rows * pi_mat.row(F_pair_coord[std::make_pair(k1, k2)]));
        W.at(k1, k2) = arma::dot(LHS, RHS);
        W.at(k2, k1) = W.at(k1, k2);
      }
    }
    // IV. Finally we compute F
    F = arma::solve(W, L.t() * D);
    // keep track of iterations
    n_updates.second += 1;
  }

private:
  arma::mat D;
  arma::mat F;
  arma::mat P;
  arma::vec q;
  arma::vec omega;
  arma::vec s;
  // K1K2 by q by N tensor
  arma::cube delta;
  // K1K2 by q matrix
  arma::mat pi_mat;
  std::map<std::pair<size_t, size_t>, size_t> F_pair_coord;
  arma::mat L;
  arma::mat W;
  // loglik
  double loglik;
  // number of threads
  int n_threads;
  // updates on the model
  std::pair<int, int> n_updates;
};
#endif
