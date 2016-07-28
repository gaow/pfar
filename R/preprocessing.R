#' @title PCA based data dimension reduction
#' @description This is part of a data preprocessing pipeline for Paired Factor Analysis (pfa) where we
#' \itemize{
#' \item perform dimension reduction on data (determine the number of components to include via Broken Stick Model)
#' \item determine centers of major clusters to get initial factors
#' \item get initial weights of factor pairs by their relative distance, e.g. via projection on principle curve
#' }
#' @param dat [N, J] data matrix
#' @param n_comp [int] number of components to keep
#' @param control \{ ... \} list of runtime variables
#' @return [N, n_comp] projected data
#' @author Gao Wang and Kushal K. Dey
#' @details
#' \itemize{
#' \item n_comp If set to NULL, a broken stick model will be applied to determine the number of PCs to include
#' }
#' @references ...
#' @examples
#' ...
#' @export
dr_pca <- function(dat, n_comp = NULL) {
  dat <- prcomp(dat)
  ## screeplot(dat, bstick = TRUE)
  if (is.null(n_comp)) {
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
    n_comp <- max(counter, 2)
  }
  n_comp <- min(ncol(dat$x), n_comp)
  return(dat$x[, 1:n_comp])
}

#' @title Initialize factors by block model
#' @description This is part of a data preprocessing pipeline for Paired Factor Analysis (pfa) where we
#' \itemize{
#' \item perform dimension reduction on data (determine the number of components to include via Broken Stick Model)
#' \item determine centers of major clusters to get initial factors
#' \item get initial weights of factor pairs by their relative distance, e.g. via projection on principle curve
#' }
#' @param dat [N, J] data matrix
#' @param n_block [int] number of blocks to learn from data
#' @param control \{ ... \} list of runtime variables
#' @return A list with elements below:
#' \item{factors}{...}
#' \item{obj}{...}
#' @author Gao Wang and Kushal K. Dey
#' @details
#' \itemize{
#' \item n_block This corresponds to the number of factors. If set to NULL, the program will not attempt to learn factors from the data.
#' }
#' @references ...
#' @examples
#' ...
#' @export
init_factor_block <- function(dat, n_block) {
  n_block <- as.integer(n_block)
  if (length(n_block) == 0 || n_block <= 0) {
    stop("Please specify number of blocks to learn from data!")
  }
  dist_mat <- as.matrix(dist(dat, method = "euclidean"))
  dat_graph <- igraph::graph.adjacency(as.matrix(dist_mat),
                                        weighted = TRUE, mode = "undirected")
  dat_edgelist <- igraph::get.edgelist(dat_graph, names = TRUE)
  signed.weight.vec <- array(0, nrow(dat_edgelist))
  for(m in 1:nrow(dat_edgelist)) {
    signed.weight.vec[m] <- dist_mat[dat_edgelist[m,1], dat_edgelist[m,2]]
  }
  weights <- max(signed.weight.vec) + 1000 - signed.weight.vec
  com_spin <- igraph::cluster_spinglass(dat_graph,
                                        weights = weights,
                                        spins = n_block)
  groups <- apply(dat, 2, function(x) tapply(x, com_spin$membership, mean))
  return(list(factors = groups, obj = com_spin))
}

#' @title Initialize weights for factor pairs by principle curve projection 
#' @description This is part of a data preprocessing pipeline for Paired Factor Analysis (pfa) where we
#' \itemize{
#' \item perform dimension reduction on data (determine the number of components to include via Broken Stick Model)
#' \item determine centers of major clusters to get initial factors
#' \item get initial weights of factor pairs by their relative distance, e.g. via projection on principle curve
#' }
#' @param dat [N, J] data matrix, used to create principle curve
#' @param factors [K, J] factor matrix whose distance is to be calculated
#' @param control \{ ... \} list of runtime variables
#' @return A list with elements below:
#' \item{factors}{...}
#' \item{weights}{...}
#' \item{lambda}{...}
#' \item{projected_points}{...}
#' @author Gao Wang and Kushal K. Dey
#' @details
#' ...
#' @references ...
#' @examples
#' ...
#' @export
init_weight_princurve <- function(dat, factors) {
  ## dat$s: projected_data_on_curve
  ## dat$lambda: lambda values
  if (is.null(dim(factors))) {
    return(NULL)
  }
  dat <- princurve::principal.curve(dat, plot = FALSE)
  factors_new <- matrix(0, dim(factors)[1], dim(factors)[2])
  projection_idx <- array(0, nrow(factors))
  for (k in 1:nrow(factors)) {
    dst <- array(0, nrow(dat$s))
    for (num in 1:length(dst)){
      dst[num] <- norm(dat$s[num,] - factors[k,], type = "2")
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
  ll <- list(weights = P,
             factors = factors_new,
             lambda = lambda,
             projected_points = dat$s)
  return(ll)
}
