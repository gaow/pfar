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
// @param tol [double_pt] tolerance for convergence
// @param maxiter [int_pt] maximum number of iterations
// @param niter [int_pt] number of iterations
// @param loglik [maxiter, 1] log likelihood, track of convergence (return)
// @param L [N, K] Loading matrix (return)
// @param status [int_pt] return status, 0 for good, 1 for error (return)
// @param logfn_1 [int_pt] log file 1 name as integer converted from character array
// @param nlf_1 [int_pt] length of above
// @param logfn_2 [int_pt] log file 2 name as integer converted from character array
// @param nlf_2 [int_pt] length of above
// @param n_threads [int_pt] number of threads for parallel processing

extern "C" int pfa_em(double *, double *, double *, double *, double *,
                      int *, int *, int *, int *, double *, int *, int *,
                      double *, double *, int *,
                      int *, int *, int *, int *, int *);

int pfa_em(double * X, double * F, double * P, double * q, double * omega,
           int * N, int * J, int * K, int * C, double * tol, int * maxiter, int * niter,
           double * loglik, double * L, int * status,
           int * logfn_1, int * nlf_1, int * logfn_2, int * nlf_2, int * n_threads) {
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
      f1_log[i] = (char) *(logfn_1 + i);
    for (int i = 0; i < *nlf_2; i++)
      f2_log[i] = (char) *(logfn_2 + i);
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
  PFA model(X, F, P, q, omega, L, *N, *J, *K, *C);
  model.set_threads(*n_threads);
  model.print(f1, 0);
  while (*niter <= *maxiter) {
    if (keeplog) {
      f1 << "#----------------------------------\n";
      f1 << "# Iteration " << *niter << "\n";
      f1 << "#----------------------------------\n";
      model.print(f1, 1);
      f2 << "#----------------------------------\n";
      f2 << "# Iteration " << *niter << "\n";
      f2 << "#----------------------------------\n";
      model.print(f2, 2);
    }
    model.get_loglik_given_nkq();
    loglik[*niter] = model.get_loglik();
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
    model.update_LF();
    model.update_weights();
  }
  if (*status)
    std::cerr << "[WARNING] EM algorithm failed to converge after " << *niter << " iterations!" << std::endl;
  if (keeplog) {
    f1.close();
    f2.close();
  }
  return 0;
}
