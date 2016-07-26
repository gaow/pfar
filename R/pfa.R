pfa <- function(X, K = NULL, F = NULL, P = NULL, q = NULL, omega = NULL, control = NULL) {
  ## Initialize data
  if (is.null(F) && is.null(K)) {
    stop("[ERROR] Please either provide K or F!")
  }
  if (is.null(F)) {
    # Initialize F with K factors all equal to the column mean of input X
    # FIXME: Have to find a smarter initilization of F
    F <- t(replicate(K, colMeans(X)))
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
    q <- (seq(0:100) - 1) / 100
  }
  if (is.null(omega)) {
    omega <- rep(1/length(q), length(q))
  }
  tol <- as.double(control$tol)
  if (length(tol) == 0 || tol <= 0) {
    tol <- 1E-6
  }
  maxiter <- as.integer(control$maxiter)
  if (length(maxiter) == 0 || maxiter <= 0) {
    maxiter <- 1000
  }
  n_cpu <- as.integer(control$n_cpu)
  if (length(n_cpu) == 0 || n_cpu <= 0) {
    n_cpu <- 1
  }
  logfile <- control$logfile
  if (is.null(logfile)) {
    f1 <- n_f1 <- f2 <- n_f2 <- 0
  } else {
    f1 <- charToRaw(paste(logfile, "result.log", sep = "."))
    f2 <- charToRaw(paste(logfile, "likelihood.log", sep = "."))
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
  return(list(F = Fout, L = Lout, P = Pout, Omega = res$omega,
              loglik = loglik, loglik_diff = diff(loglik),
              niter = res$niter, status = res$status))
}
