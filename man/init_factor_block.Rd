% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/preprocessing.R
\name{init_factor_block}
\alias{init_factor_block}
\title{Initialize factors by block model}
\usage{
init_factor_block(dat, n_block)
}
\arguments{
\item{dat}{[N, J] data matrix}

\item{n_block}{[int] number of blocks to learn from data}

\item{control}{\{ ... \} list of runtime variables}
}
\value{
A list with elements below:
\item{factors}{...}
\item{obj}{...}
}
\description{
This is part of a data preprocessing pipeline for Paired Factor Analysis (pfa) where we
\itemize{
\item perform dimension reduction on data (determine the number of components to include via Broken Stick Model)
\item determine centers of major clusters to get initial factors
\item get initial weights of factor pairs by their relative distance, e.g. via projection on principle curve
}
}
\details{
\itemize{
\item n_block This corresponds to the number of factors. If set to NULL, the program will not attempt to learn factors from the data.
}
}
\examples{
...
}
\references{
...
}
\author{
Gao Wang and Kushal K. Dey
}
