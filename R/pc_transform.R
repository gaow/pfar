#' @title PCA based data transformation and factor initialization for pfa
#' @description This is a data preprocessing pipeline for Paired Factor Analysis (pfa) where we
#' \itemize{
#' \item perform PCA on data
#' \item determine the number of PCs to include via Broken Stick Model
#' \item determine centers of major clusters from the PC space via block model
#' \item project these centers to principle curve to produce initial values of factors and their weights for pfa
#' }
#' @param dat [N, J] data matrix
#' @param n_pc [int] number of PC's to keep
#' @param control \{spins = 3, auto_factor = TRUE\} list of runtime variables
#' @return A list with elements below:
#' \item{data}{...}
#' \item{F}{...}
#' \item{P}{...}
#' @author Gao Wang and Kushal K. Dey
#' @details
#' \itemize{
#' \item n_pc: if set to NULL, a broken stick model will be applied to determine the number of PCs to include
#' }
#' @references ...
#' @examples
#' ...
#' @export
pc_transform <- function(dat, n_pc = NULL, control = NULL) {
  dat <- pca(dat, n_pc)
  projection <- NULL
  if (is.null(control$auto_factor) || control$auto_factor) {
    factors <- block_analysis(dat, control$spins)$groups
    if (nrow(factors) >= 2) {
      projection <- princurve_projection(dat, factors)
    }
  }
  return(list(data = dat, F = projection$F, P = projection$P))
}

## PCA analysis with broken stick model for number of PCs
pca <- function(dat, n_pc = NULL) {
  dat <- prcomp(dat)
  ## screeplot(dat, bstick = TRUE)
  if (is.null(n_pc)) {
    bstick_var <- vegan::bstick(dat)
    prcomp_var <- dat$sdev^2
    counter <- 0
    for(i in 1:length(bstick_var)) {
      if (bstick_var[i] > prcomp_var[i]){
        break
      } else {
        counter <- counter + 1
      }
    }
    n_pc <- max(counter, 2)
  }
  n_pc <- min(ncol(dat$x), n_pc)
  return(dat$x[, 1:n_pc])
}

## Block model
block_analysis <- function(dat, spins = NULL) {
  spins <- as.integer(spins)
  if (length(spins) == 0 || spins <= 0) {
    spins <- 3
  }
  dist_pc <- as.matrix(dist(dat, method = "euclidean"))
  dat_graph <- igraph::graph.adjacency(as.matrix(dist_pc),
                                        weighted = TRUE, mode = "undirected")
  dat_edgelist <- igraph::get.edgelist(dat_graph, names = TRUE)
  signed.weight.vec <- array(0, nrow(dat_edgelist))
  for(m in 1:nrow(dat_edgelist)) {
    signed.weight.vec[m] <- dist_pc[dat_edgelist[m,1], dat_edgelist[m,2]]
  }
  weights <- max(signed.weight.vec) + 1000 - signed.weight.vec
  com_spin <- igraph::cluster_spinglass(dat_graph,
                                        weights = weights,
                                        spins = spins);
  groups <- apply(dat, 2, function(x) tapply(x, com_spin$membership, mean))
  return(list(groups = groups, com_spin = com_spin))
}


## Principle curve projection
princurve_projection <- function(dat, factors) {
  dat <- princurve::principal.curve(dat, plot = FALSE)
  ## dat$s: projected_data_on_curve
  ## dat$lambda: lambda values
  factors_new <- matrix(0, dim(factors)[1], dim(factors)[2])
  projection_idx <- array(0, nrow(factors))
  for (k in 1:nrow(factors)) {
    dst <- array(0, nrow(dat$s))
    for (num in 1:length(dst)){
      dst[num] <- norm(dat$s[num,] - factors[k,], type="2")
    }
    factors_new[k,] <- dat$s[which.min(dst),]
    projection_idx[k] <- which.min(dst)
  }
  lambda <- dat$lambda[projection_idx]
  P <- matrix(0, nrow(factors), nrow(factors))
  for (k1 in 1:nrow(factors)) {
    for (k2 in 1:k1) {
        if (k2 < k1) {
            P[k1, k2] <- 1/(abs(lambda[k1] - lambda[k2]))
        }
    }
  }
  P <- P/sum(P)
  ll <- list(P = P,
             F = factors_new,
             lambda = lambda,
             projected_points = dat$s)
  return(ll)
}
