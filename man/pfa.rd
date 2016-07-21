\name{pfa}
\alias{pfa}
\title{Paired factor analysis model}
\description{...}

\usage{
pfa(X)
}

\arguments{
  \item{X}{[N, J] matrix of observed data}
  \item{K}{[int] number of factors}
  \item{F}{[K, J] initial factor matrix}
  \item{P}{[K, K] initial factor pair frequency matrix}
  \item{q}{[C, 1] initial vector of possible membership loadings, a discrete set}
  \item{omega}{[C, 1] initial weight of membership loadings, a discrete set corresponding to q}
}

\value{
  \item{loglik}{log likelihood after convergence}
}

\details{
  ...
}

\author{Gao Wang and Kushal K. Dey}

\references{
  ...
}

\examples{
library("pfar")
# ?pfa
dat = readRDS('vignettes/example_data.rds')
pfa(dat$X, K = 15)
}