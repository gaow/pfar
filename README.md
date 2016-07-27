This repository contains an R package for performing paired factor analysis.

To install the `pfar` package,
```
devtools::install_github("gaow/pfar")
```
If you do not have `devtools` or have no internet access, you need to [obtain the source code](https://github.com/gaow/pfar/archive/master.zip), decompress the tarball and type `make` to install the package.

## Running paired factor analysis

The main function in the `pfar` is `pfa`:
```
> ?pfar::pfa
```
You can follow the `example` section of the documentation to run the program on an example data set.

## Troubleshoot

### Linear algebra support
If you get error message *Cannot find lapack / blas* you need to install `LAPAC` and `BLAS` libraries. On Debian Linux:
```
sudo apt-get install libblas-dev liblapack-dev
```
### OpenMP
`pfar` requires compilers with OpenMP support. On Debian Linux:
```
sudo apt-get install libgomp1
```
