pfa <- function(X, K = NULL, F = NULL, P = NULL, q = NULL, omega = NULL, controls = NULL) {
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
  tol <- controls$tol
  if (is.null(tol) || tol <= 0) {
    tol <- 1E-6
  }
  maxiter <- controls$maxiter
  if (is.null(maxiter) || maxiter <= 5) {
    maxiter <- 1000
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
            as.double(as.vector(omega)),
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
            PACKAGE = "pfar")
  ## Process output
  Fout <- matrix(res$F, nrow(F), ncol(F))
  Lout <- matrix(res$L, nrow(L), ncol(L))
  Pout <- matrix(res$P, nrow(P), ncol(P))
  track_c <- res$track_c[1:(res$niter + 1)]
  return(list(F = Fout, L = Lout, P = Pout, track_c = track_c, loglik_diff = diff(track_c),
              niter = res$niter, status = res$status))
}
