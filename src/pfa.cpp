// Gao Wang and Kushal K. Dey (c) 2016
// code format configuration: clang-format -style=Google -dump-config >
// ~/.clang-format
#include "pfa.hpp"
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>

//! EM algorithm for paired factor analysis
// @param X [N, J] observed data matrix
// @param F [K, J] initial factor matrix
// @param P [K, K] initial frequency matrix of factor pairs, an upper diagonal
// matrix
// @param q [C, 1] initial vector of possible membership loadings, a discrete
// set. It has to be ordered (ascending), start with zero and end with a value
// smaller than one.
// @param N [int_pt] number of rows of matrix X
// @param J [int_pt] number of columns of matrix X and F
// @param K [int_pt] number of rows of matrix F and P
// @param C [int_pt] number of elements in q
// @param alpha0 [double_pt] Dirichlet prior for factor weights
// @param variational [int_pt] 0 or 1, whether or not to use variational method
// @param tol [double_pt] tolerance for convergence
// @param maxiter [int_pt] maximum number of iterations
// @param niter [int_pt] number of iterations
// @param loglik [maxiter, 1] log likelihood, track of convergence (return)
// @param L [N, K] Loading matrix (return)
// @param alpha [K, K] Dirichlet posterior parameter matrix for factor pair
// weights (return)
// (return)
// @param status [int_pt] return status, 0 for good, 1 for error (return)
// @param logfn_1 [int_pt] log file 1 name as integer converted from character
// array
// @param nlf_1 [int_pt] length of above
// @param logfn_2 [int_pt] log file 2 name as integer converted from character
// array
// @param nlf_2 [int_pt] length of above
// @param n_threads [int_pt] number of threads for parallel processing

int pfa_em(double* X, double* F, double* P, double* q, int* N, int* J, int* K,
           int* C, double* alpha0, int* variational, double* tol, int* maxiter,
           int* niter, double* loglik, double* L, double* alpha, int* status,
           int* logfn_1, int* nlf_1, int* logfn_2, int* nlf_2, int* n_threads) {
  //
  // Set up logfiles
  //
  std::fstream f1;
  std::fstream f2;
  if (*nlf_1 > 0) {
    char f1_log[(*nlf_1) + 1];
    char f2_log[(*nlf_2) + 1];
    for (int i = 0; i < *nlf_1; i++) f1_log[i] = (char)*(logfn_1 + i);
    for (int i = 0; i < *nlf_2; i++) f2_log[i] = (char)*(logfn_2 + i);
    f1_log[*nlf_1] = '\0';
    f2_log[*nlf_2] = '\0';
    // log file
    f1.open(f1_log, std::fstream::out);
    time_t now;
    time(&now);
    f1 << "#\n# " << asctime(localtime(&now)) << "#\n\n";
    f1.close();
    f1.open(f1_log, std::fstream::app);
    // debug file
    if (*nlf_2 > 0) {
      f2.open(f2_log, std::fstream::out);
      f2 << "#\n# " << asctime(localtime(&now)) << "#\n\n";
      f2.close();
      f2.open(f2_log, std::fstream::app);
    }
  }
  //
  // Fit model
  //
  *niter = 0;
  PFA* model;
  if (*variational)
    model = new PFA_VEM(X, F, P, q, L, alpha, *N, *J, *K, *C, *alpha0);
  else
    model = new PFA_EM(X, F, P, q, L, *N, *J, *K, *C);
  model->set_threads(*n_threads);
  model->write(f1, 0);
  while (*niter <= *maxiter) {
    if (f1.is_open()) {
      f1 << "#----------------------------------\n";
      f1 << "# Iteration " << *niter << "\n";
      f1 << "#----------------------------------\n";
      model->write(f1, 1);
      if (f2.is_open()) {
        f2 << "#----------------------------------\n";
        f2 << "# Iteration " << *niter << "\n";
        f2 << "#----------------------------------\n";
        model->write(f2, 2);
      }
    }
    int e_status = model->E_step();
    if (e_status != 0) {
      std::cerr << "[ERROR] E step failed!" << std::endl;
      *status = 1;
      break;
    }
    loglik[*niter] = model->get_loglik();
    if (loglik[*niter] != loglik[*niter]) {
      std::cerr << "[ERROR] likelihood nan produced!" << std::endl;
      *status = 1;
      break;
    }
    if (f1.is_open()) {
      f1 << "Loglik:\n" << loglik[*niter] << "\n";
    }
    (*niter)++;
    // check convergence
    if (*niter > 1) {
      double diff = loglik[(*niter) - 1] - loglik[(*niter) - 2];
      // check monotonicity
      if (diff < 0.0) {
        std::cerr << "[ERROR] likelihood decreased in EM algorithm!"
                  << std::endl;
        *status = 1;
        break;
      }
      // converged
      if (diff < *tol) break;
    }
    if (*niter == *maxiter) {
      // did not converge
      *status = 1;
      break;
    }
    // continue with more iterations
    model->M_step();
  }
  if (*status)
    std::cerr << "[WARNING] PFA failed to converge at tolerance level " << *tol
              << " after " << *niter << " iterations!" << std::endl;
  if (f1.is_open()) f1.close();
  if (f2.is_open()) f2.close();
  return 0;
}

//! Model likelihood for PFA
// @param D [N, J] matrix of simulated data
// @param F [K, J] true factor matrix
// @param P [K, K] true factor frequency matrix
// @param Q [N, 3] true position of samples
// @param S [J, 1] vector of simulated standard deviation for features
// @param N [int_pt] number of rows of matrix X
// @param J [int_pt] number of columns of matrix X and F
// @param K [int_pt] number of rows of matrix F and P
// @param loglik [double_pt] log likelihood (return)
int pfa_model_loglik(double* D, double* F, double* P, double* Q, double* S,
                     int* N, int* J, int* K, double* loglik) {
  // fake data to initialize PFA class that will not get used
  double* q, *L;
  q = (double*)malloc(sizeof(double));
  L = (double*)malloc(sizeof(double) * (*N) * (*K));
  PFA model = PFA(D, F, P, q, L, *N, *J, *K, 1);
  arma::vec true_s(S, *J, false, true);
  arma::mat true_q(Q, *N, 3, false, true);
  model.update_model_loglik(true_s, true_q);
  *loglik = model.get_loglik();
  return 0;
}

// this computes log delta
// return: k1k2 by q by N tensor of log delta
void PFA::update_ldelta(int core) {
#pragma omp parallel for num_threads(n_threads)
  for (size_t k1 = 0; k1 < F.n_rows; k1++) {
    for (size_t k2 = 0; k2 <= k1; k2++) {
      // given k1, k2 and q, density is a N-vector
      for (size_t qq = 0; qq < q.n_elem; qq++) {
        if ((k1 != k2) | (k1 == k2 & qq == 0)) {
          arma::vec Dn_delta = arma::zeros<arma::vec>(D.n_rows);
          for (size_t j = 0; j < D.n_cols; j++) {
            arma::vec density = D.col(j);
            double m =
                (k2 < k1)
                    ? q.at(qq) * F.at(k2, j) + (1 - q.at(qq)) * F.at(k1, j)
                    : F.at(k1, j);
            Dn_delta += density.transform(
                [=](double x) { return (normal_pdf_log(x, m, s.at(j))); });
          }
          if (k1 != k2) Dn_delta += std::log(1 / node_fudge);
          if (core == 0) Dn_delta += std::log(P.at(k1, k2));
          // FIXME: this is slow, due to the cube/slice structure
          for (size_t n = 0; n < D.n_rows; n++)
            delta.slice(n).at(F_pair_coord[std::make_pair(k1, k2)], qq) =
                Dn_delta.at(n);
        }
      }
    }
  }
  ldelta = delta;
}

void PFA_VEM::update_variational_ldelta() {
  delta = ldelta;
#pragma omp parallel for num_threads(n_threads)
  for (size_t k1 = 0; k1 < F.n_rows; k1++) {
    for (size_t k2 = 0; k2 <= k1; k2++) {
      // FIXME: this is slow, due to the cube/slice structure
      for (size_t n = 0; n < D.n_rows; n++) {
        delta.slice(n).row(F_pair_coord[std::make_pair(k1, k2)]) +=
            digamma_alpha.at(k1, k2) - digamma_sum_alpha;
      }
    }
  }
}

// this computes loglik and delta
// this results in a N by k1k2 matrix of loglik
// and the delta tensor on its original (exp) scale, with each slice summing to
// one
void PFA::update_loglik_and_delta() {
  arma::vec loglik_vec;
  loglik_vec.set_size(D.n_rows);
#pragma omp parallel for num_threads(n_threads)
  for (size_t n = 0; n < D.n_rows; n++) {
    // 1. find the log of sum of exp(delta.slice(n))
    // 2. exp transform delta to its proper scale: set delta prop to exp(delta)
    // 3. scale delta.slice(n) to sum to 1
    // a numeric trick is used to calculate log(sum(exp(x)))
    // lsum=function(lx){
    //  m = max(lx)
    //  m + log(sum(exp(lx-m)))
    // }
    // see gaow/pfar/issue/2 for a discussion
    // FIXME: A hack for k == k single factor case, part 1
    for (size_t k = 0; k < F.n_rows; k++) {
      // reset single factor case: remove zeros
      delta.slice(n).row(F_pair_coord[std::make_pair(k, k)]).fill(
          delta.slice(n).at(F_pair_coord[std::make_pair(k, k)], 0));
    }
    double delta_n_max = delta.slice(n).max();
    delta.slice(n) = arma::exp(delta.slice(n) - delta_n_max);
    // FIXME: A hack for k == k single factor case, part 2
    for (size_t k = 0; k < F.n_rows; k++) {
      // reset single factor case: has to be zero for all but the first grid
      double tmp = delta.slice(n).at(F_pair_coord[std::make_pair(k, k)], 0);
      delta.slice(n).row(F_pair_coord[std::make_pair(k, k)]).fill(0);
      delta.slice(n).at(F_pair_coord[std::make_pair(k, k)], 0) = tmp;
    }
    double sum_delta_n = arma::accu(delta.slice(n));
    loglik_vec.at(n) = std::log(sum_delta_n) + delta_n_max;
    delta.slice(n) = delta.slice(n) / sum_delta_n;
  }
  loglik = arma::accu(loglik_vec);
}

// Factors F and loadings L are to be updated here
void PFA::update_factor_model() {
  // Need to compute 2 matrices in order to solve F
  L.fill(0);
  arma::mat L2 = L;
#pragma omp parallel for num_threads(n_threads) collapse(2)
  for (size_t k = 0; k < F.n_rows; k++) {
    // I. First we compute the k-th column for E(L), the N X K matrix:
    // generate the proper input for N X Q %*% Q X 1
    // where the N X Q matrix is taken from delta given K1K2
    // and the Q X 1 matrix correspond to the grid of q
    // II. Then we compute the diagonal elements for E(W), the K X K matrix
    // First we still compute a N X K matrix like above but replacing q / 1 - q
    // with q^2 / (1 - q)^2
    // then take colsum to get the K vector as the diagonal
    for (size_t i = 0; i < F.n_rows; i++) {
      // create the LHS Dk1k2, N X Q matrix from delta given k1, k2
      arma::mat Dk1k2(D.n_rows, q.n_elem, arma::fill::zeros);
      for (size_t n = 0; n < D.n_rows; n++) {
        Dk1k2.row(n) = delta.slice(n).row(F_pair_coord[std::make_pair(k, i)]);
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
          L.col(k) += Dk1k2.col(0);
          L2.col(k) += Dk1k2.col(0);
        }
      }
    }
  }
  // II. diagonal elements for E(W)
  W.diag() = arma::sum(L2);
  // III. Compute off-diagonal elements for E(W), the K X K matrix
  // for k1 != k2
  // it involves on the LHS a vector of [q1(1-q1), q2(1-q2) ...]
  // and on the RHS for each pair of (k1, k2) the corresponding row from delta
  // summing over samples
  arma::vec LHS = q % (1 - q);
  arma::mat RHS = arma::sum(delta, 2);
  RHS = RHS.t();
#pragma omp parallel for num_threads(n_threads)
  for (size_t k1 = 0; k1 < F.n_rows; k1++) {
    for (size_t k2 = 0; k2 < k1; k2++) {
      W.at(k1, k2) =
          arma::dot(LHS, RHS.col(F_pair_coord[std::make_pair(k1, k2)]));
      W.at(k2, k1) = W.at(k1, k2);
    }
  }
  // IV. Solve F
  F = arma::solve(W, L.t() * D);
}

// S is the residual standard error vector to be updated for each feature
// FIXME: can this be optimized via transposing the tensor delta?
void PFA::update_residual_error() {
#pragma omp parallel for num_threads(n_threads)
  for (size_t j = 0; j < D.n_cols; j++) {
    s.at(j) = 0;
    for (size_t k1 = 0; k1 < F.n_rows; k1++) {
      for (size_t k2 = 0; k2 <= k1; k2++) {
        for (size_t n = 0; n < D.n_rows; n++) {
          if (k2 < k1) {
            for (size_t qq = 0; qq < q.n_elem; qq++) {
              s.at(j) +=
                  delta.slice(n).at(F_pair_coord[std::make_pair(k1, k2)], qq) *
                  std::pow(D.at(n, j) - q.at(qq) * F.at(k2, j) -
                               (1 - q.at(qq)) * F.at(k1, j),
                           2);
            }
          } else {
            s.at(j) +=
                delta.slice(n).at(F_pair_coord[std::make_pair(k1, k2)], 0) *
                std::pow(D.at(n, j) - F.at(k1, j), 2);
          }
        }
      }
    }
    s.at(j) = std::sqrt(s.at(j) / double(D.n_rows));
  }
}

// update factor pair weights
void PFA_EM::update_paired_factor_weights() {
  arma::vec pik1k2 = arma::vectorise(arma::mean(arma::sum(delta, 1), 2));
#pragma omp parallel for num_threads(n_threads)
  for (size_t k1 = 0; k1 < P.n_rows; k1++) {
    for (size_t k2 = 0; k2 <= k1; k2++) {
      P.at(k1, k2) = pik1k2.at(F_pair_coord[std::make_pair(k1, k2)]);
    }
  }
}

// update variational parameters
void PFA_VEM::update_variational_parameters() {
#pragma omp parallel for num_threads(n_threads)
  for (size_t k1 = 0; k1 < P.n_rows; k1++) {
    for (size_t k2 = 0; k2 <= k1; k2++) {
      double tmp = 0;
      arma::vec tmp_vec = arma::vectorise(arma::sum(arma::sum(delta, 2), 0));
      tmp = alpha0 + tmp_vec.at(F_pair_coord[std::make_pair(k1, k2)]);
      alpha.at(k1, k2) = tmp;
      digamma_alpha.at(k1, k2) = digamma(tmp);
    }
  }
  digamma_sum_alpha = digamma(arma::accu(alpha));
}

void PFA_VEM::update_paired_factor_weights() { P = alpha / arma::accu(alpha); }

double PFA_VEM::get_variational_lowerbound() {
  double lowerbound = loglik;
  // add prior part for factor weights
  for (size_t k1 = 0; k1 < P.n_rows; k1++) {
    for (size_t k2 = 0; k2 <= k1; k2++) {
      lowerbound += (alpha.at(k1, k2) - 1) * std::log(P.at(k1, k2));
    }
  }
  return lowerbound;
}
