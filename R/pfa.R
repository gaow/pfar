pfa <- function(X, K, F = NULL, P = NULL, q = NULL, omega = NULL) {
  ## Initialize data
  if (is.null(F)) {
    # Initialize F with K factors all equal to the column mean of input X
    # FIXME: Have to find a smarter initilization of F
    F <- t(replicate(K, colMeans(X)))
  }
  if (is.null(P)) {
    # Initialize P uniformly
    P <- matrix(2 / ((K + 1) * K), K, K)
  }
  if (is.null(q)) {
    # Initialze q to be 0/100, 1/100, 2/100, ..., 1
    q <- (seq(0:100) - 1) / 100
  }
  if (is.null(omega)) {
    omega <- rep(1/length(q), length(q))
  }
  ## sanity check
  stopifnot(nrow(F) == K)
  stopifnot(ncol(X) == ncol(F))
  stopifnot(nrow(F) == nrow(P))
  stopifnot(nrow(P) == ncol(P))
  stopifnot(length(q) == length(omega))
  ## data processing
  P[upper.tri(P)] <- 0
  ## factor analysis
  res <- .C("pfa_em",
            as.double(as.vector(X)),
            as.double(as.vector(F)),
            as.double(as.vector(P)),
            as.double(as.vector(q)),
            as.double(as.vector(omega)),
            as.integer(nrow(X)),
            as.integer(ncol(X)),
            as.integer(nrow(F)),
            as.integer(length(q)),
            PACKAGE = "pfar")
  return(NULL)
}
