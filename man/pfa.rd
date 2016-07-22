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
  \item{control}{\{tol = 1E-6, maxiter = 1000, logfile = NULL\} list of runtime variables}
}

\value{
  \item{F}{...}
  \item{L}{...}
  \item{P}{...}
  \item{Omega}{...}
  \item{track_c}{...}
  \item{loglik_diff}{...}
  \item{niter}{...}
  \item{status}{...}
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
control = list(logfile = 'example_data.pfa')
pfa(dat$X, K = 15, control = control)
}