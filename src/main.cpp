#include <armadillo>
#include <iostream>
int main() {
  double av[10] = { 5, 6, 7, 8, 5, 5, 5, 5, 5, 5 };
  arma::mat A(av, 5, 2, false, true);
  A.print();
  return 0;
}
