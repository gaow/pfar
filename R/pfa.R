pfa <- function(X, K, F = NULL, P = NULL, q = NULL, omega = NULL, controls = NULL) {
  ## Initialize data
  if (is.null(F)) {
    # Initialize F with K factors all equal to the column mean of input X
    # FIXME: Have to find a smarter initilization of F
    F <- t(replicate(K, colMeans(X)))
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
  ## sanity check
  stopifnot(nrow(F) == K)
  stopifnot(ncol(X) == ncol(F))
  stopifnot(nrow(F) == nrow(P))
  stopifnot(nrow(P) == ncol(P))
  stopifnot(length(q) == length(omega))
  ## factor analysis
  loglik <- 0
  niter <- 0
  L <- matrix(0, nrow(X), nrow(F))
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
            niter = as.integer(niter),
            loglik = as.integer(loglik),
            L = as.double(as.vector(L)),
            PACKAGE = "pfar")
  # FIXME: need to reshape
  return(list(loglik = res$loglik, niter = res$niter))
}
