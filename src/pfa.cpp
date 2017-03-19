// Gao Wang and Kushal K. Dey (c) 2016
#include <iostream>
#include <fstream>
#include <cstdio>
#include <ctime>
#include "pfa.hpp"

//! EM algorithm for paired factor analysis
// @param X [N, J] observed data matrix
// @param F [K, J] initial factor matrix
// @param P [K, K] initial frequency matrix of factor pairs, an upper diagonal matrix
// @param q [C, 1] initial vector of possible membership loadings, a discrete set
// @param N [int_pt] number of rows of matrix X
// @param J [int_pt] number of columns of matrix X and F
// @param K [int_pt] number of rows of matrix F and P
// @param C [int_pt] number of elements in q
// @param alpha0 [double_pt] Dirichlet prior for factor weights
// @param beta0 [double_pt] Dirichlet prior for grid weights
// @param tol [double_pt] tolerance for convergence
// @param maxiter [int_pt] maximum number of iterations
// @param niter [int_pt] number of iterations
// @param loglik [maxiter, 1] log likelihood, track of convergence (return)
// @param L [N, K] Loading matrix (return)
// @param alpha [K, K] Dirichlet posterior parameter matrix for factor pair weights (return)
// @param beta [C, 1] Dirichlet posterior parameter vector for grid weights (return)
// @param status [int_pt] return status, 0 for good, 1 for error (return)
// @param logfn_1 [int_pt] log file 1 name as integer converted from character array
// @param nlf_1 [int_pt] length of above
// @param logfn_2 [int_pt] log file 2 name as integer converted from character array
// @param nlf_2 [int_pt] length of above
// @param n_threads [int_pt] number of threads for parallel processing

int pfa_em(double * X, double * F, double * P, double * q,
           int * N, int * J, int * K, int * C, double * alpha0, double * beta0,
           double * tol, int * maxiter, int * niter,
           double * loglik, double * L, double * alpha, double * beta,
           int * status, int * logfn_1, int * nlf_1, int * logfn_2, int * nlf_2, int * n_threads)
{
	//
	// Set up logfiles
	//
	bool keeplog = (*nlf_1 > 0) ? true : false;
	std::fstream f1;
	std::fstream f2;

	if (keeplog) {
		char f1_log[(*nlf_1) + 1];
		char f2_log[(*nlf_2) + 1];
		for (int i = 0; i < *nlf_1; i++)
			f1_log[i] = (char)*(logfn_1 + i);
		for (int i = 0; i < *nlf_2; i++)
			f2_log[i] = (char)*(logfn_2 + i);
		f1_log[*nlf_1] = '\0';
		f2_log[*nlf_2] = '\0';
		f1.open(f1_log, std::fstream::out);
		f2.open(f2_log, std::fstream::out);
		time_t now;
		time(&now);
		f1 << "#\n# " << asctime(localtime(&now)) << "#\n\n";
		f2 << "#\n# " << asctime(localtime(&now)) << "#\n\n";
		f1.close();
		f2.close();
		f1.open(f1_log, std::fstream::app);
		f2.open(f2_log, std::fstream::app);
	}
	//
	// Fit model
	//
	*niter = 0;
	PFA_EM model(X, F, P, q, L, *N, *J, *K, *C);
	model.set_threads(*n_threads);
	model.write(f1, 0);
	while (*niter <= *maxiter) {
		if (keeplog) {
			f1 << "#----------------------------------\n";
			f1 << "# Iteration " << *niter << "\n";
			f1 << "#----------------------------------\n";
			model.write(f1, 1);
			f2 << "#----------------------------------\n";
			f2 << "# Iteration " << *niter << "\n";
			f2 << "#----------------------------------\n";
			model.write(f2, 2);
		}
		int e_status = model.E_step();
    if (e_status != 0) {
      std::cerr << "[ERROR] E step failed!" << std::endl;
      *status = 1;
			break;
    }
		loglik[*niter] = model.get_loglik();
    if (loglik[*niter] != loglik[*niter]) {
      std::cerr << "[ERROR] likelihood nan produced!" << std::endl;
			*status = 1;
			break;
    }
		if (keeplog) {
			f1 << "Loglik:\t" << loglik[*niter] << "\n";
		}
		(*niter)++;
		// check convergence
		if (*niter > 1) {
			double diff = loglik[(*niter) - 1] - loglik[(*niter) - 2];
			// check monotonicity
			if (diff < 0.0) {
				std::cerr << "[ERROR] likelihood decreased in EM algorithm!" << std::endl;
				*status = 1;
				break;
			}
			// converged
			if (diff < *tol)
				break;
		}
		if (*niter == *maxiter) {
			// did not converge
			*status = 1;
			break;
		}
		// continue with more iterations
		model.M_step();
	}
	if (*status)
		std::cerr << "[WARNING] PFA failed to converge after " << *niter << " iterations!" <<
		std::endl;
	if (keeplog) {
		f1.close();
		f2.close();
	}
	return 0;
}

// this computes delta
// return: N by k1k2 matrix of log delta
void PFA_EM::update_ldelta() {
#pragma omp parallel for num_threads(n_threads)
  for (size_t k1 = 0; k1 < F.n_rows; k1++) {
    for (size_t k2 = 0; k2 <= k1; k2++) {
      // given k1, k2 and q, density is a N-vector
      arma::vec Dn_delta = arma::zeros<arma::vec>(D.n_rows);
      for (size_t qq = 0; qq < q.n_elem; qq++) {
        if ((k1 != k2) | (k1 == k2 & qq == 0)) {
          for (size_t j = 0; j < D.n_cols; j++) {
            arma::vec density = D.col(j);
            double m = (k2 < k1) ? q.at(qq) * F.at(k2, j) + (1 - q.at(qq)) * F.at(k1, j) : F.at(k1, j);
            Dn_delta += density.transform( [=](double x) { return (normal_pdf_log(x, m, s.at(j))); } );
          }
        }
      }
      delta.col(F_pair_coord[std::make_pair(k1, k2)]) = Dn_delta + std::log(P.at(k1, k2));
    }
  }
}

// this computes loglik and delta
// this results in a N by k1k2 matrix of loglik
// and a N by k1k2 matrix of delta
void PFA_EM::update_loglik_and_delta() {
  arma::vec loglik_vec;
  loglik_vec.set_size(D.n_rows);
#pragma omp parallel for num_threads(n_threads)
  for (size_t n = 0; n < D.n_rows; n++) {
    // 1. find the log of sum of exp(delta.row(n))
    // 2. exp transform delta to its proper scale: set delta prop to exp(delta)
    // 3. scale delta to sum to 1
    // a numeric trick is used to calculate log(sum(exp(x)))
    // lsum=function(lx){
    //  m = max(lx)
    //  m + log(sum(exp(lx-m)))
    // }
    // see gaow/pfar/issue/2 for a discussion
    double delta_n_max = delta.row(n).max();
    delta.row(n) = arma::exp(delta.row(n) - delta_n_max);
    double sum_delta_n = arma::accu(delta.row(n));
    loglik_vec.at(n) = std::log(sum_delta_n) + delta_n_max;
    delta.row(n) = delta.row(n) / sum_delta_n;
  }
  loglik = arma::accu(loglik_vec);
}

// update factor pair weights
void PFA_EM::update_paired_factor_weights() {
  arma::vec pik1k2 = arma::vectorise(arma::mean(delta));
#pragma omp parallel for num_threads(n_threads)
  for (size_t k1 = 0; k1 < P.n_rows; k1++) {
    for (size_t k2 = 0; k2 <= k1; k2++) {
      P.at(k1, k2) = pik1k2.at(F_pair_coord[std::make_pair(k1, k2)]);
    }
  }
}

void PFA_EM::update_factor_model() {
  // Factors F and loadings L are to be updated here
  // Need to compute 2 matrices in order to solve F
  L.fill(0);
  arma::mat L2 = L;
#pragma omp parallel for num_threads(n_threads) collapse(2)
  for (size_t k = 0; k < F.n_rows; k++) {
    // I. Compute the k-th column for E(L), the N X K matrix:
    // and II. the diagonal elements for E(W), the K X K matrix
    for (size_t i = 0; i < F.n_rows; i++) {
#pragma omp critical
      {
      if (k < i) {
        // I.
        L.col(k) += delta.col(F_pair_coord[std::make_pair(k, i)]) * avg_q;
        // II.
        L2.col(k) += delta.col(F_pair_coord[std::make_pair(k, i)]) * avg_q2;
      }
      if (k > i) {
        L.col(k) += delta.col(F_pair_coord[std::make_pair(k, i)]) * avg_1q;
        L2.col(k) += delta.col(F_pair_coord[std::make_pair(k, i)]) * avg_1q2;
      }
      if (k == i) {
        L.col(k) += delta.col(F_pair_coord[std::make_pair(k, i)]);
        L2.col(k) += delta.col(F_pair_coord[std::make_pair(k, i)]);
      }
      }
    }
  }
  // II. diagonal elements for E(W)
  W.diag() = arma::sum(L2);
  // III. Compute off-diagonal elements for E(W), the K X K matrix
#pragma omp parallel for num_threads(n_threads)
  for (size_t k1 = 0; k1 < F.n_rows; k1++) {
    for (size_t k2 = 0; k2 < k1; k2++) {
      W.at(k1, k2) = arma::sum(delta.col(F_pair_coord[std::make_pair(k1, k2)]) * avg_q1q);
      W.at(k2, k1) = W.at(k1, k2);
    }
  }
  // IV. Solve F
  F = arma::solve(W, L.t() * D);
}

// S is the residual standard error vector to be updated for each feature
void PFA_EM::update_residual_error() {
#pragma omp parallel for num_threads(n_threads)
  for (size_t j = 0; j < D.n_cols; j++) {
    s.at(j) = 0;
    for (size_t k1 = 0; k1 < F.n_rows; k1++) {
      for (size_t k2 = 0; k2 <= k1; k2++) {
        if (k2 < k1) {
          arma::vec tmp = arma::zeros<arma::vec>(D.n_rows);
          for (size_t qq = 0; qq < q.n_elem; qq++) {
            tmp += arma::pow(D.col(j) - q.at(qq) * F.at(k2, j) - (1 - q.at(qq)) * F.at(k1, j), 2);
          }
          s.at(j) += arma::accu(delta.col(F_pair_coord[std::make_pair(k1, k2)]) / double(q.n_elem) % tmp);
        } else {
          s.at(j) += arma::accu(delta.col(F_pair_coord[std::make_pair(k1, k2)]) % arma::pow(D.col(j) - F.at(k1, j), 2));
        }
      }
    }
    s.at(j) = std::sqrt(s.at(j) / double(D.n_rows));
  }
}
