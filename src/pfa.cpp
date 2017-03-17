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
// @param omega [C, 1] initial weight of membership loadings, a discrete set corresponding to q
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

extern "C" int pfa_em(double *, double *, double *, double *, double *,
                      int *, int *, int *, int *, double *, double *,
                      double *, int *, int *,
                      double *, double *, double *, double *,
                      int *, int *, int *, int *, int *, int *);

int pfa_em(double * X, double * F, double * P, double * q, double * omega,
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
	// Fit model via EM
	//
	*niter = 0;
	PFA model(X, F, P, q, omega, L, alpha, beta, *N, *J, *K, *C, *alpha0, *beta0);
	model.set_threads(*n_threads);
  model.set_variational();
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
		int variational_status = model.fit();
    if (variational_status != 0) {
      std::cerr << "[ERROR] variational inference procedure failed!" << std::endl;
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
		model.update();
	}
	if (*status)
		std::cerr << "[WARNING] EM algorithm failed to converge after " << *niter << " iterations!" <<
		std::endl;
	if (keeplog) {
		f1.close();
		f2.close();
	}
	return 0;
}

void PFA_EM::update_loglik() {
  // this computes loglik and delta
  // this results in a N by k1k2 matrix of loglik
  // and a N by k1k2 matrix of delta
#pragma omp parallel for num_threads(n_threads)
  for (size_t k1 = 0; k1 < F.n_rows; k1++) {
    for (size_t k2 = 0; k2 <= k1; k2++) {
      // given k1, k2 and q, density is a N-vector
      // in the end of the loop, the vector should populate a column of a slice of the tensor
      arma::vec Dn_delta = arma::zeros<arma::vec>(D.n_rows);
      bool update = true
      for (size_t qq = 0; qq < q.n_elem; qq++) {
        if (k1 == k2 & qq > 0) {
          update = false;
          break;
        }
        for (size_t j = 0; j < D.n_cols; j++) {
          arma::vec density = D.col(j);
          double m = (k2 < k1) ? q.at(qq) * F.at(k2, j) + (1 - q.at(qq)) * F.at(k1, j) : F.at(k1, j);
          Dn_delta += density.transform( [=](double x) { return (normal_pdf_log(x, m, s.at(j))); } );
        }
      }
      if (update)
        delta.col(F_pair_coord[std::make_pair(k1, k2)]) = Dn_delta + std::log(P.at(k1, k2));
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
    double delta_n_max = delta.row(n).max();
    delta.row(n) = arma::exp(delta.row(n) - delta_n_max);
    double sum_delta_n = arma::accu(delta.row(n));
    loglik_vec.at(n) = std::log(sum_delta_n) + delta_n_max;
    delta.row(n) = delta.row(n) / sum_delta_n;
  }
  loglik = arma::accu(loglik_vec);
}

void PFA_EM::update_weights() {
  // update factor weights
  arma::vec pik1k2 = delta.mean();
#pragma omp parallel for num_threads(n_threads)
  for (size_t k1 = 0; k1 < P.n_rows; k1++) {
    for (size_t k2 = 0; k2 <= k1; k2++) {
      P.at(k1, k2) = pik1k2.at(F_pair_coord[std::make_pair(k1, k2)]);
    }
  }
}

void PFA_EM::update_LFS() {
  // Factors F and loadings L are to be updated here
  // Need to compute 2 matrices in order to solve F
  // S is the residual standard error vector to be updated for each feature
  L.fill(0);
  arma::mat L2 = L;
  arma::mat cum_delta_paired = arma::sum(delta_paired, 2);
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
      arma::vec RHS = arma::vectorise(cum_delta_paired.row(F_pair_coord[std::make_pair(k1, k2)]));
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
