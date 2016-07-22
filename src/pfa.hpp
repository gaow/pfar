#ifndef _PFA_HPP
#define _PFA_HPP

#include <armadillo>
#include <map>

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
    has_F_pair_coord = false;
    L.set_size(D.n_rows, F.n_rows);
    W.set_size(F.n_rows, F.n_rows);
    log_delta.set_size(D.n_rows, int((F.n_rows - 1) * F.n_rows / 2), q.n_elem);
    avg_delta.set_size(int((P.n_rows - 1) * P.n_rows / 2), q.n_elem);
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
      log_delta.print(out, "log(delta) tensor:");
      avg_delta.print(out, "delta averaged over samples:");
    }
  }

  void get_log_delta_given_nkq() {
    // this computes delta up to a normalizing constant
    // this results in a N by k1k2 by q tensor of loglik
    for (size_t qq = 0; qq < q.n_elem; qq++) {
      size_t col_cnt = 0;
      for (size_t k1 = 0; k1 < F.n_rows; k1++) {
        for (size_t k2 = 0; k2 < k1; k2++) {
          // given k1, k2 and q, density is a N-vector
          // in the end of the loop, the vector should populate a column of a slice of the tensor
          arma::vec Dn_llik = arma::zeros<arma::vec>(D.n_rows);
          for (size_t j = 0; j < D.n_cols; j++) {
            arma::vec density = D.col(j);
            double m = q.at(qq) * F.at(k2, j) + (1 - q.at(qq)) * F.at(k1, j);
            double sig = s.at(j);
            Dn_llik += density.transform( [=](double x) { return (normal_pdf_log(x, m, sig)); } );
          }
          Dn_llik = Dn_llik + (std::log(P.at(k1, k2)) + std::log(omega.at(qq)));
          // populate the tensor
          log_delta.slice(qq).col(col_cnt) = Dn_llik;
          // set factor pair coordinates in the tensor
          if (!has_F_pair_coord) {
            F_pair_coord[std::make_pair(k1, k2)] = col_cnt;
            F_pair_coord[std::make_pair(k2, k1)] = col_cnt;
          }
          col_cnt++;
        }
        has_F_pair_coord = true;
      }
    }
  }

  double get_loglik_prop() {
    // log likelihood up to a normalizing constant
    return arma::accu(log_delta);
  }

  void update_weights() {
    // update P and omega
    // this results in a K1K2 by q matrix of avglik
    // which equals pi_k1k2 * omega_q
    arma::cube slice_rowsums = arma::sum(arma::exp(log_delta));
    for (size_t qq = 0; qq < q.n_elem; qq++) {
      avg_delta.col(qq) = arma::vectorise(slice_rowsums.slice(qq)) / D.n_rows;
    }
    // sum over q grids
    arma::vec pik1k2 = arma::sum(avg_delta, 1);
    pik1k2 = pik1k2 / arma::sum(pik1k2);
    size_t col_cnt = 0;
    for (size_t k1 = 0; k1 < P.n_rows; k1++) {
      for (size_t k2 = 0; k2 < k1; k2++) {
        P.at(k1, k2) = pik1k2.at(col_cnt);
        col_cnt++;
      }
    }
    // sum over (k1, k2)
    omega = arma::vectorise(arma::sum(avg_delta));
    omega = omega / arma::sum(omega);
  }

  void update_F() {
    // F, the K by J matrix, is to be updated here
    // Need to compute 2 matrices in order to solve F
    // The loading, L is N X K matrix; W = L'L is K X K matrix
    L.fill(0);
    W.fill(0);
    for (size_t k = 0; k < F.n_rows; k++) {
      // I. First we compute the k-th column for E(L), the N X K matrix:
      // generate the proper input for 1 X K %*% K X N
      // where the 1 X K matrix is loadings Lk consisting of q or (1-q)
      // and the K X N matrix is the delta for a corresponding subset of a slice from delta
      // II. Then we compute the diagonal elements for E(W), the K X K matrix
      // Because we need to sum over all N and we have computed this before,
      // we can work with avg_delta (K1K2 X q) instead of log_delta the tensor
      // we need to loop over the q slices
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
        // from a slice of the tensor log_delta
        // where the rows are N's, the columns corresponds to
        // k1k2, k1k3, k2k3, k1k4, k2k4, k3k4, k1k5 ... along the lower triangle matrix P
        // use F_pair_coord to get proper index for data from the tensor
        arma::mat Lk2(D.n_rows, F.n_rows, arma::fill::zeros);
        for (size_t i = 0; i < F.n_rows; i++) {
          if (k != i) Lk2.col(i) = arma::exp(log_delta.slice(qq).col(F_pair_coord[std::make_pair(k, i)]));
        }
        // Update the k-th column of L
        L.col(k) += arma::vectorise(Lk1 * Lk2.t());
        // II. ........................................
        for (size_t i = 0; i < F.n_rows; i++) {
          if (i < k) Lk1.at(0, i) = (1.0 - qi) * (1.0 - qi);
          if (i > k) Lk1.at(0, i) = qi * qi;
        }
        arma::mat Lk3(F.n_rows, 1, arma::fill::zeros);
        for (size_t i = 0; i < F.n_rows; i++) {
          if (k != i) Lk3.at(i, 0) = D.n_rows * avg_delta.at(F_pair_coord[std::make_pair(k, i)], qq);
        }
        // Update E(W_kk)
        W.at(k, k) += arma::as_scalar(Lk1 * Lk3);
      }
    }
    // III. Now we compute off-diagonal elements for E(W), the K X K matrix
    // it involves on the LHS a vector of [q1(1-q1), q2(1-q2) ...]
    // and on the RHS for each pair of (k1, k2) the corresponding row from avg_delta
    arma::vec LHS = q.transform( [](double val) { return (val * (1.0 - val)); } );
    for (size_t k1 = 0; k1 < F.n_rows; k1++) {
      for (size_t k2 = 0; k2 < k1; k2++) {
        arma::vec RHS = arma::vectorise(D.n_rows * avg_delta.row(F_pair_coord[std::make_pair(k1, k2)]));
        W.at(k1, k2) = arma::dot(LHS, RHS);
        W.at(k2, k1) = W.at(k1, k2);
      }
    }
    // IV. Finally we compute F
    F = arma::solve(W, L.t() * D);
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
  std::map<std::pair<int,int>, int> F_pair_coord;
  bool has_F_pair_coord;
  arma::mat L;
  arma::mat W;
};
#endif
