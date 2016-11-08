// Gao Wang and Kushal K. Dey (c) 2016
#ifndef _PFA_HPP
#define _PFA_HPP

#include <armadillo>
#include <map>
#include <omp.h>

static const double INV_SQRT_2PI = 0.3989422804014327;
static const double INV_SQRT_2PI_LOG = -0.91893853320467267;

inline double normal_pdf(double x, double m, double sd)
{
  double a = (x - m) / sd;
  return INV_SQRT_2PI / sd * std::exp(-0.5 * a * a);
};

inline double normal_pdf_log(double x, double m, double sd)
{
  double a = (x - m) / sd;
  return INV_SQRT_2PI_LOG - std::log(sd) -0.5 * a * a;
};

template <typename T>
void print(const T& e) { std::cout << e << std::endl; }

class PFA {
public:
  PFA(double * cX, double * cF, double * cP, double * cQ, double * omega,
      double * cLout, int N, int J, int K, int C, double alpha, double beta):
    // mat(aux_mem*, n_rows, n_cols, copy_aux_mem = true, strict = true)
    D(cX, N, J, false, true), F(cF, K, J, false, true),
    P(cP, K, K, false, true), q(cQ, C, false, true),
    omega(omega, C, false, true), L(cLout, N, K, false, true),
    alpha(alpha), beta(beta)
  {
    // initialize residual sd with sample sd
    s = arma::vectorise(arma::stddev(D));
    L.set_size(D.n_rows, F.n_rows);
    W.set_size(F.n_rows, F.n_rows);
    delta_paired.set_size(int((F.n_rows - 1) * F.n_rows / 2), q.n_elem, D.n_rows);
    delta_single.set_size(D.n_rows, F.n_rows);
    pi_paired.set_size(int((P.n_rows - 1) * P.n_rows / 2), q.n_elem);
    pi_single.set_size(P.n_rows);
    n_threads = 1;
    n_updates = 0;
    // set factor pair coordinates to avoid
    // having to compute it at each iteration
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

  void write(std::ostream& out, int info) {
    if (info == 0) {
      // D.print(out, "Data Matrix:");
      q.print(out, "Membership grids:");
      omega.print(out, "Membership grid weight:");
    }
    if (info == 1) {
      F.print(out, "Factor matrix:");
      P.print(out, "Factor frequency matrix:");
      L.print(out, "Loading matrix:");
    }
    if (info == 2) {
      if (n_updates > 1) {
        pi_paired.print(out, "Joint weight for factor pairs and membership grid:");
        pi_single.print(out, "Joint weight for single factors and membership grid:");
      }
      // W.print(out, "E[L'L] matrix:");
      s.print(out, "Residual standard deviation of data columns:");
    }
  }

  void set_threads(int n) {
    n_threads = n;
  }

  void get_loglik_given_nkq() {
    // this computes loglik and delta
    // this results in a N by k1k2 by q tensor of loglik
    // and a k1k2 by q by N tensor of delta_paired
    // and a N by K matrix of delta_single
    // and a k1k2 by q matrix of pi_paired
    // and a k vector of pi_single
#pragma omp parallel for num_threads(n_threads)
    for (size_t qq = 0; qq < q.n_elem; qq++) {
      for (size_t k1 = 0; k1 < F.n_rows; k1++) {
        for (size_t k2 = 0; k2 <= k1; k2++) {
          // given k1, k2 and q, density is a N-vector
          // in the end of the loop, the vector should populate a column of a slice of the tensor
          arma::vec Dn_delta = arma::zeros<arma::vec>(D.n_rows);
          for (size_t j = 0; j < D.n_cols; j++) {
            arma::vec density = D.col(j);
            double m = (k2 < k1) ? q.at(qq) * F.at(k2, j) + (1 - q.at(qq)) * F.at(k1, j) : F.at(k1, j);
            Dn_delta += density.transform( [=](double x) { return (normal_pdf_log(x, m, s.at(j))); } );
          }
          // populate the delta tensor
          if (n_updates == 0) {
            // No update on pi is available yet: approximate it under independence assumption
            // Dn_delta on the log scale
            Dn_delta = Dn_delta + std::log(P.at(k1, k2)) + std::log(omega.at(qq));
          }
          else {
            Dn_delta = Dn_delta + ((k2 < k1) ?
                                   std::log(pi_paired.at(F_pair_coord[std::make_pair(k1, k2)], qq)) :
                                   std::log(pi_single.at(k1)));
          }
          if (k2 < k1) {
            // FIXME: this is slow, due to the cube/slice structure
            for (size_t n = 0; n < D.n_rows; n++) {
              delta_paired.slice(n).at(F_pair_coord[std::make_pair(k1, k2)], qq) = Dn_delta.at(n);
            }
          } else {
            delta_single.col(k1) = Dn_delta;
          }
        }
      }
    }
    // Compute log likelihood and pi
    arma::vec loglik_vec;
    loglik_vec.set_size(D.n_rows);
#pragma omp parallel for num_threads(n_threads)
    for (size_t n = 0; n < D.n_rows; n++) {
      // scale delta to avoid very small likelihood driving the product to zero
      // and take exp so that it goes to normal scale
      // see github issue 1 for details
      // a numeric trick is used to calculate log(sum(exp(x)))
      double delta_n_max = std::max(delta_paired.slice(n).max(), delta_single.row(n).max());
      delta_paired.slice(n) = arma::exp(delta_paired.slice(n) - delta_n_max);
      delta_single.row(n) = arma::exp(delta_single.row(n) - delta_n_max);
      double sum_delta_n = arma::accu(delta_paired.slice(n)) + arma::accu(delta_single.row(n));
      loglik_vec.at(n) = std::log(sum_delta_n) + delta_n_max;
      delta_paired.slice(n) = delta_paired.slice(n) / sum_delta_n;
      delta_single.row(n) = delta_single.row(n) / sum_delta_n;
    }
    pi_paired = arma::mean(delta_paired, 2);
    pi_single = arma::mean(delta_single, 1);
    loglik = arma::accu(loglik_vec);
  }

  double get_loglik() {
    // current log likelihood
    return loglik;
  }

  void update_weights() {
    // update factor weights sum over q grids
    arma::vec pik1k2 = arma::sum(pi_paired, 1);
    pik1k2 = pik1k2 / arma::sum(pik1k2);
#pragma omp parallel for num_threads(n_threads)
    for (size_t k1 = 0; k1 < P.n_rows; k1++) {
      for (size_t k2 = 0; k2 <= k1; k2++) {
        if (k2 < k1)
          P.at(k1, k2) = pik1k2.at(F_pair_coord[std::make_pair(k1, k2)]);
        else
          P.at(k1, k1) = pi_single.at(k1);
      }
    }
  }

  void update_LFS() {
    // Factors F and loadings L are to be updated here
    // Need to compute 2 matrices in order to solve F
    // S is the residual standard error vector to be updated for each feature
    L.fill(0);
    arma::mat L2 = L;
#pragma omp parallel for num_threads(n_threads) collapse(2)
    for (size_t k = 0; k < F.n_rows; k++) {
      // I. First we compute the k-th column for E(L), the N X K matrix:
      // generate the proper input for N X C %*% C X 1
      // where the N X C matrix is taken from delta given K1K2
      // and the C X 1 matrix correspond to the grid of q
      // II. Then we compute the diagonal elements for E(W), the K X K matrix
      // First we still compute a N X K matrix like above but replacing q / 1 - q with q^2 / (1 - q)^2
      // then take colsum to get the K vector as the diagonal
      //
      // 0. (prepare)
      for (size_t i = 0; i < F.n_rows; i++) {
        // create the LHS Dk1k2, N X C matrix from delta given k1, k2
        arma::mat Dk1k2(D.n_rows, q.n_elem, arma::fill::zeros);
        for (size_t n = 0; n < D.n_rows; n++) {
          Dk1k2.row(n) = delta_paired.slice(n).row(F_pair_coord[std::make_pair(k, i)]);
        }
#pragma omp critical
        {
        if (k < i) {
          // I.
          L.col(k) += Dk1k2 * q;
          // II.
          L2.col(k) += Dk1k2 * (q % q);
        }
        if (k > i) {
          L.col(k) += Dk1k2 * (1 - q);
          L2.col(k) += Dk1k2 * ((1 - q) % (1 - q));
        }
        if (k == i) {
          L.col(k) += delta_single.col(k);
          L2.col(k) += delta_single.col(k);
        }
        }
      }
    }
    // II. diagonal elements for E(W)
    W.diag() = arma::sum(L2);
    // III. Now we compute off-diagonal elements for E(W), the K X K matrix
    // it involves on the LHS a vector of [q1(1-q1), q2(1-q2) ...]
    // and on the RHS for each pair of (k1, k2) the corresponding row from pi_mat
    arma::vec LHS = q % (1 - q);
#pragma omp parallel for num_threads(n_threads)
    for (size_t k1 = 0; k1 < F.n_rows; k1++) {
      for (size_t k2 = 0; k2 < k1; k2++) {
        arma::vec RHS = arma::vectorise(D.n_rows * pi_paired.row(F_pair_coord[std::make_pair(k1, k2)]));
        W.at(k1, k2) = arma::dot(LHS, RHS);
        W.at(k2, k1) = W.at(k1, k2);
      }
    }
    // IV. Finally we compute F
    F = arma::solve(W, L.t() * D);
    // V. ... and update s
#pragma omp parallel for num_threads(n_threads)
    // FIXME: can this be optimized? perhaps hard unless we re-design data structure
    for (size_t j = 0; j < D.n_cols; j++) {
      s.at(j) = 0;
      for (size_t qq = 0; qq < q.n_elem; qq++) {
        for (size_t k1 = 0; k1 < F.n_rows; k1++) {
          for (size_t k2 = 0; k2 <= k1; k2++) {
            if (k2 < k1) {
              for (size_t n = 0; n < D.n_rows; n++) {
                s.at(j) += delta_paired.slice(n).at(F_pair_coord[std::make_pair(k1, k2)], qq) * std::pow(D.at(n, j) - q.at(qq) * F.at(k2, j) - (1 - q.at(qq)) * F.at(k1, j), 2);
              }
            } else {
              s.at(j) += arma::accu(delta_single.col(k1) % arma::pow(D.col(j) - F.at(k1, j), 2));
            }
          }
        }
      }
      s.at(j) = std::sqrt(s.at(j) / double(D.n_rows));
    }
    // keep track of iterations
    n_updates += 1;
  }

private:
  // N by J matrix of data
  arma::mat D;
  // K by J matrix of factors
  arma::mat F;
  // K by K matrix of factor pair frequencies
  arma::mat P;
  // Q by 1 vector of membership grids
  arma::vec q;
  // Q by 1 vector of membership grid weights
  arma::vec omega;
  // J by 1 vector of residual standard error
  arma::vec s;
  // K1K2 by Q by N tensor
  arma::cube delta_paired;
  // N by K matrix
  arma::mat delta_single;
  // K1K2 by Q matrix
  arma::mat pi_paired;
  // K vector
  arma::vec pi_single;
  std::map<std::pair<size_t, size_t>, size_t> F_pair_coord;
  // N by K matrix of loadings
  arma::mat L;
  // W = L'L is K X K matrix
  arma::mat W;
  // loglik
  double loglik;
  // number of threads
  int n_threads;
  // updates on the model
  int n_updates;
  // Dirichlet priors for factors and grids
  double alpha;
  double beta;
};
#endif
