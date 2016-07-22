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
  tol <- control$tol
  if (is.null(tol) || tol <= 0) {
    tol <- 1E-6
  }
  maxiter <- control$maxiter
  if (is.null(maxiter) || maxiter <= 0) {
    maxiter <- 1000
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
  track_c <- rep(-999, maxiter)
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
            track_c = as.double(as.vector(track_c)),
            L = as.double(as.vector(L)),
            status = as.integer(status),
            as.integer(as.vector(f1)),
            as.integer(n_f1),
            as.integer(as.vector(f2)),
            as.integer(n_f2),
            PACKAGE = "pfar")
  ## Process output
  Fout <- matrix(res$F, nrow(F), ncol(F))
  Lout <- matrix(res$L, nrow(L), ncol(L))
  Pout <- matrix(res$P, nrow(P), ncol(P))
  track_c <- res$track_c[1:res$niter]
  return(list(F = Fout, L = Lout, P = Pout, Omega = res$omega,
              track_c = track_c, loglik_diff = diff(track_c),
              niter = res$niter, status = res$status))
}
