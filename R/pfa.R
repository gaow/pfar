pfa <- function(X) {
  res <- .C("pfa_em",
            as.double(as.vector(X)),
            as.integer(dim(X)[1]),
            as.integer(dim(X)[2]),
            PACKAGE = "pfar")
  return(res)
}
