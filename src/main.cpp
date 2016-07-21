#include "pfa.hpp"
int main() {
  double av[10] = { 5, 6, 7, 8, 5, 5, 5, 5, 5, 5 };
  arma::mat model(av, 5, 2, false, true);
  model.print();
  arma::cube s = arma::randu<arma::cube>(2,3,2);
  arma::cube s1 = sum(s);
  s.print();
  s1.print();
  return 0;
}
