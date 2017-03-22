// Gao Wang and Kushal K. Dey (c) 2016
// code format configuration: clang-format -style=Google -dump-config >
// ~/.clang-format
#ifndef _PFA_HPP
#define _PFA_HPP

#include <omp.h>
#include <armadillo>
#include <map>
#include <string>

static const double INV_SQRT_2PI = 0.3989422804014327;
static const double INV_SQRT_2PI_LOG = -0.91893853320467267;

inline double normal_pdf(double x, double m, double sd) {
  double a = (x - m) / sd;
  return INV_SQRT_2PI / sd * std::exp(-0.5 * a * a);
};

inline double normal_pdf_log(double x, double m, double sd) {
  double a = (x - m) / sd;
  return INV_SQRT_2PI_LOG - std::log(sd) - 0.5 * a * a;
};

/* The digamma function is the derivative of gammaln.
   Reference:
    J Bernardo,
    Psi ( Digamma ) Function,
    Algorithm AS 103,
    Applied Statistics,
    Volume 25, Number 3, pages 315-317, 1976.
    From http://www.psc.edu/~burkardt/src/dirichlet/dirichlet.f
    (with modifications for negative numbers and extra precision)
*/
inline double digamma(double x) {
  double neginf = -INFINITY;
  static const double c = 12, digamma1 = -0.57721566490153286,
                      trigamma1 = 1.6449340668482264365, /* pi^2/6 */
      s = 1e-6, s3 = 1. / 12, s4 = 1. / 120, s5 = 1. / 252, s6 = 1. / 240,
                      s7 = 1. / 132, s8 = 691. / 32760, s9 = 1. / 12,
                      s10 = 3617. / 8160;
  double result;
  /* Illegal arguments */
  if ((x == neginf) || std::isnan(x)) {
    return NAN;
  }
  /* Singularities */
  if ((x <= 0) && (floor(x) == x)) {
    return neginf;
  }
  /* Negative values */
  /* Use the reflection formula (Jeffrey 11.1.6):
 * digamma(-x) = digamma(x+1) + pi*cot(pi*x)
 *
 * This is related to the identity
 * digamma(-x) = digamma(x+1) - digamma(z) + digamma(1-z)
 * where z is the fractional part of x
 * For example:
 * digamma(-3.1) = 1/3.1 + 1/2.1 + 1/1.1 + 1/0.1 + digamma(1-0.1)
 *               = digamma(4.1) - digamma(0.1) + digamma(1-0.1)
 * Then we use
 * digamma(1-z) - digamma(z) = pi*cot(pi*z)
 */
  if (x < 0) {
    return digamma(1 - x) + M_PI / tan(-M_PI * x);
  }
  /* Use Taylor series if argument <= S */
  if (x <= s) return digamma1 - 1 / x + trigamma1 * x;
  /* Reduce to digamma(X + N) where (X + N) >= C */
  result = 0;
  while (x < c) {
    result -= 1 / x;
    x++;
  }
  /* Use de Moivre's expansion if argument >= C */
  /* This expansion can be computed in Maple via asympt(Psi(x),x) */
  if (x >= c) {
    double r = 1 / x, t;
    result += log(x) - 0.5 * r;
    r *= r;
#if 1
    result -= r * (s3 - r * (s4 - r * (s5 - r * (s6 - r * s7))));
#else
    /* this version for lame compilers */
    t = (s5 - r * (s6 - r * s7));
    result -= r * (s3 - r * (s4 - r * t));
#endif
  }
  return result;
}

template <typename T>
void print(const T &e) {
  std::cout << e << std::endl;
}

class Exception {
 public:
  /// constructor
  /// \param msg error message
  Exception(const std::string &msg) : m_msg(msg) {}

  /// return error message
  const char *message() { return m_msg.c_str(); }

  virtual ~Exception(){};

 private:
  /// error message
  std::string m_msg;
};

/// exception, thrown if out of memory
class StopIteration : public Exception {
 public:
  StopIteration(const std::string msg) : Exception(msg){};
};

/// exception, thrown if index out of range
class IndexError : public Exception {
 public:
  IndexError(const std::string msg) : Exception(msg){};
};

/// exception, thrown if value of range etc
class ValueError : public Exception {
 public:
  ValueError(const std::string msg) : Exception(msg){};
};

/// exception, thrown if system error occurs
class SystemError : public Exception {
 public:
  SystemError(const std::string msg) : Exception(msg){};
};

/// exception, thrown if a runtime error occurs
class RuntimeError : public Exception {
 public:
  RuntimeError(const std::string msg) : Exception(msg){};
};

extern "C" int pfa_em(double *, double *, double *, double *, int *, int *,
                      int *, int *, double *, int *, double *, int *, int *,
                      double *, double *, double *, int *, int *, int *, int *,
                      int *, int *);

class PFA {
 public:
  PFA(double *cX, double *cF, double *cP, double *cQ, double *cLout, int N,
      int J, int K, int C)
      :  // mat(aux_mem*, n_rows, n_cols, copy_aux_mem = true, strict = true)
        D(cX, N, J, false, true),
        F(cF, K, J, false, true),
        P(cP, K, K, false, true),
        q(cQ, C, false, true),
        L(cLout, N, K, false, true) {
    // initialize residual sd with sample sd
    s = arma::vectorise(arma::stddev(D));
    W.set_size(F.n_rows, F.n_rows);
    delta.set_size(int((F.n_rows + 1) * F.n_rows / 2), q.n_elem, D.n_rows);
    delta.fill(0);
    n_threads = 1;
    n_updates = 0;
    node_fudge = 1.0;
    for (size_t k1 = 0; k1 < F.n_rows; k1++) {
      for (size_t k2 = 0; k2 <= k1; k2++) {
        // set factor pair coordinates to avoid
        // having to compute it at each iteration
        // (b + 1) * b / 2 - (b - a)
        size_t k1k2 = size_t(k1 * (k1 + 1) / 2 + k2);
        F_pair_coord[std::make_pair(k1, k2)] = k1k2;
        F_pair_coord[std::make_pair(k2, k1)] = k1k2;
      }
    }
  }
  virtual ~PFA() {}

  virtual PFA *clone() const { return new PFA(*this); }

  virtual void write(std::ostream &out, int info) {
    throw RuntimeError("The base write() function should not be called");
  }
  virtual int E_step() {
    throw RuntimeError("The base E_step() function should not be called");
  }

  virtual int M_step() {
    throw RuntimeError("The base M_step() function should not be called");
  }

  void set_threads(int n) { n_threads = n; }
  void update_ldelta(int core = 0);
  void update_loglik_and_delta();
  void update_factor_model();
  void update_residual_error();
  double get_loglik() { return loglik; }

 protected:
  // N by J matrix of data
  arma::mat D;
  // K by K matrix of factor pair frequencies
  arma::mat P;
  // K by J matrix of factors
  arma::mat F;
  // Q by 1 vector of membership grids
  arma::vec q;
  // J by 1 vector of residual standard error
  arma::vec s;
  // K1K2 by Q by N tensor
  arma::cube delta;
  arma::cube ldelta;
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
  // fudge factor for more weights on the node
  double node_fudge;
};

class PFA_EM : public PFA {
 public:
  PFA_EM(double *cX, double *cF, double *cP, double *cQ, double *cLout, int N,
         int J, int K, int C)
      : PFA(cX, cF, cP, cQ, cLout, N, J, K, C) {}

  PFA *clone() const { return new PFA_EM(*this); }

  void update_paired_factor_weights();
  int E_step() {
    update_ldelta();
    update_loglik_and_delta();
    update_paired_factor_weights();
    return 0;
  }

  int M_step() {
    update_factor_model();
    update_residual_error();
    n_updates += 1;
    return 0;
  }

  void write(std::ostream &out, int info) {
    if (info == 0) {
      // D.print(out, "Data Matrix:");
      q.print(out, "Membership grids:");
    }
    if (info == 1) {
      F.print(out, "Factor matrix:");
      P.print(out, "Factor frequency matrix:");
      if (n_updates > 0)
        L.print(out, "Loading matrix:");
      else
        out << "Loading matrix:" << std::endl;
    }
    if (info == 2) {
      // W.print(out, "E[L'L] matrix:");
      s.print(out, "Residual standard deviation of data columns:");
      if (n_updates > 0) {
        ldelta.print(out, "log delta tensor");
        delta.print(out, "delta tensor");
      }
    }
  }
};

class PFA_VEM : public PFA {
 public:
  PFA_VEM(double *cX, double *cF, double *cP, double *cQ, double *cLout,
          double *calphaout, int N, int J, int K, int C, double alpha0)
      : PFA(cX, cF, cP, cQ, cLout, N, J, K, C),
        alpha(calphaout, K, K, false, true),
        alpha0(alpha0) {
    alpha.zeros();
    digamma_alpha.set_size(P.n_rows, P.n_rows);
    digamma_alpha.fill(digamma(alpha0));
    digamma_sum_alpha = 0;
    maxiter = 1000;
    tol = 1E-3;
  }

  PFA *clone() const { return new PFA_VEM(*this); }

  void update_variational_ldelta();
  double get_variational_lowerbound();
  void update_variational_parameters();
  void update_paired_factor_weights();

  int E_step() {
    int status = 0;
    update_ldelta(1);
    size_t niter = 0;
    lowerbound.resize(0);
    while (niter <= maxiter) {
      update_variational_ldelta();
      update_loglik_and_delta();
      update_variational_parameters();
      update_paired_factor_weights();
      lowerbound.push_back(get_variational_lowerbound());
      if (lowerbound.back() != lowerbound.back()) {
        std::cerr << "[ERROR] lower bound nan produced in variational approximation!"
                  << std::endl;
        status = 1;
        break;
      }
      niter++;
      if (niter > 1) {
        double diff = lowerbound[niter - 1] - lowerbound[niter - 2];
        if (diff < 0.0) {
          std::cerr << "[ERROR] lower bound decreased in variational "
                       "approximation:  \n\tfrom "
                    << lowerbound[niter - 2] << " to " << lowerbound[niter - 1]
                    << "!" << std::endl;
          status = 1;
          break;
        }
        if (diff < tol) break;
      }
      if (niter == maxiter) {
        std::cerr << "[WARNIKNG] variational approximation procedure failed to "
                     "converge at tolerance level "
                  << tol << ", after " << maxiter
                  << " iterations: \n\tlog lower bound starts "
                  << lowerbound.front() << ", ends " << lowerbound.back() << "!"
                  << std::endl;
        status = 1;
        break;
      }
    }
    return status;
  }

  int M_step() {
    update_factor_model();
    update_residual_error();
    n_updates += 1;
    return 0;
  }

  void write(std::ostream &out, int info) {
    if (info == 0) {
      // D.print(out, "Data Matrix:");
      q.print(out, "Membership grids:");
    }
    if (info == 1) {
      F.print(out, "Factor matrix:");
      P.print(out, "Factor frequency matrix:");
      if (n_updates > 0)
        L.print(out, "Loading matrix:");
      else
        out << "Loading matrix:" << std::endl;
    }
    if (info == 2) {
      // W.print(out, "E[L'L] matrix:");
      s.print(out, "Residual standard deviation of data columns:");
      if (n_updates > 0) {
        alpha.print(out, "Dirichlet parameter for factor pairs:");
        out << "log lower bound:" << std::endl;
        for (size_t i = 0; i < lowerbound.size(); ++i) {
          out << lowerbound[i] << ", ";
        }
        out << std::endl;
      }
    }
  }

 private:
  // Dirichlet priors for factors and grids
  double alpha0;
  // K by K matrix of digamma of variational parameter for factor pair
  // frequencies
  arma::mat digamma_alpha;
  arma::mat alpha;
  double digamma_sum_alpha;
  size_t maxiter;
  double tol;
  std::vector<double> lowerbound;
};
#endif
