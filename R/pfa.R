#' @title Paired factor analysis model
#' @description ...
#' @param X [N, J] matrix of observed data
#' @param K [int] number of factors
#' @param F [K, J] initial factor matrix
#' @param P [K, K] initial factor pair frequency matrix
#' @param q [C, 1] initial vector of possible membership loadings, a discrete set
#' @param omega [C, 1] initial weight of membership loadings, a discrete set corresponding to q
#' @param control \{tol = 1E-5, maxiter = 10000, logfile = NULL, n_cpu = 1\} list of runtime variables
#' @return A list with elements below:
#' \item{F}{...}
#' \item{L}{...}
#' \item{P}{...}
#' \item{omega}{...}
#' \item{loglik}{...}
#' \item{loglik_diff}{...}
#' \item{niter}{...}
#' \item{status}{...}
#' @details ...
#' @author Gao Wang and Kushal K. Dey
#' @references ...
#' @examples
#' ## The example data set can be installed via
#' ## devtools::install_github("kkdey/singleCellRNASeqMouseDeng2014")
#' library(singleCellRNASeqMouseDeng2014)
#' meta_data <- pData(Deng2014MouseESC)
#' dat = exprs(Deng2014MouseESC)
#' K = 6
#' dat = t(limma::voom(dat)$E)
#' init_val = pfar::init_weight_princurve(dat, pfar::init_factor_block(pfar::dr_pca(dat), K))
#' control = list(logfile = 'example_data.pfa', n_cpu = 8)
#' res = pfar::pfa(dat, F = init_val$factors, P = init_val$weights, control = control)
#' print(res)
#' @useDynLib pfar
#' @export

pfa <- function(X, K = NULL, F = NULL, P = NULL, q = NULL, omega = NULL, control = NULL) {
  ## Initialize data
  if (is.null(F) && is.null(K)) {
    stop("[ERROR] Please provide either K or F!")
  }
  if (is.null(F)) {
    # Initialize F with random K factors
    # FIXME: Have to find a smarter initilization of F
    F <- X[sample(nrow(X), size = K, replace = FALSE),]
  } else {
    K <- nrow(F)
  }
  if (is.null(P)) {
    # Initialize P uniformly
    P <- matrix(0, K, K)
    P[lower.tri(P)] <- 2 / ((K - 1) * K)
  }
  if (is.null(q)) {
    # Initialze q to be 0/100, 1/100, 2/100, ..., 1
    q <- seq(0, 100) / 100
  }
  if (is.integer(q)) {
    q <- seq(0, 100, by = max(as.integer(100 / q), 1)) / 100
  }
  if (is.null(omega)) {
    omega <- rep(1/length(q), length(q))
  }
  tol <- as.double(control$tol)
  if (length(tol) == 0 || tol <= 0) {
    tol <- 1E-4
  }
  maxiter <- as.integer(control$maxiter)
  if (length(maxiter) == 0 || maxiter <= 0) {
    maxiter <- 10000
  }
  n_cpu <- as.integer(control$n_cpu)
  if (length(n_cpu) == 0 || n_cpu <= 0) {
    n_cpu <- 1
  }
  logfile <- control$logfile
  if (is.null(logfile)) {
    f1 <- n_f1 <- f2 <- n_f2 <- 0
  } else {
    f1 <- charToRaw(paste(logfile, "updates.log", sep = "."))
    f2 <- charToRaw(paste(logfile, "debug.log", sep = "."))
    n_f1 <- length(f1)
    n_f2 <- length(f2)
  }
  ## sanity check
  stopifnot(nrow(F) == K)
  stopifnot(ncol(X) == ncol(F))
  stopifnot(nrow(F) == nrow(P))
  stopifnot(nrow(P) == ncol(P))
  stopifnot(length(q) == length(omega))
  ## factor analysis
  loglik <- rep(-999, maxiter)
  niter <- 0
  L <- matrix(0, nrow(X), nrow(F))
  status <- 0
  res <- .C("pfa_em",
            as.double(as.vector(X)),
            F = as.double(as.vector(F)),
            P = as.double(as.vector(P)),
            as.double(as.vector(q)),
            omega = as.double(as.vector(omega)),
            as.integer(nrow(X)),
            as.integer(ncol(X)),
            as.integer(nrow(F)),
            as.integer(length(q)),
            as.double(tol),
            as.integer(maxiter),
            niter = as.integer(niter),
            loglik = as.double(as.vector(loglik)),
            L = as.double(as.vector(L)),
            status = as.integer(status),
            as.integer(as.vector(f1)),
            as.integer(n_f1),
            as.integer(as.vector(f2)),
            as.integer(n_f2),
            as.integer(n_cpu),
            PACKAGE = "pfar")
  ## Process output
  Fout <- matrix(res$F, nrow(F), ncol(F))
  Lout <- matrix(res$L, nrow(L), ncol(L))
  Pout <- matrix(res$P, nrow(P), ncol(P))
  loglik <- res$loglik[1:res$niter]
  return(list(F_init = F, F = Fout, L = Lout, P = Pout, omega = res$omega,
              loglik = loglik, loglik_diff = diff(loglik),
              niter = res$niter, status = res$status))
}
