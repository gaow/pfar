#include "pfa.hpp"
int main() {
  double av[10] = { 5, 6, 7, 8, 5, 5, 5, 5, 5, 5 };
  arma::mat model(av, 5, 2, false, true);
  model.print();
  return 0;
}
