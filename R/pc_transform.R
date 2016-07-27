#' @title PCA based data transformation and factor initialization for pfa
#' @description This is a data preprocessing pipeline for Paired Factor Analysis (pfa) where we
#' \itemize{
#' \item perform PCA on data
#' \item determine the number of PCs to include via Broken Stick Model
#' \item determine centers of major clusters from the PC space via block model
#' \item project these centers to principle curve to produce initial values of factors and their weights for pfa
#' }
#' @author Gao Wang and Kushal K. Dey
#' @references ...
#' @examples
#' @export
