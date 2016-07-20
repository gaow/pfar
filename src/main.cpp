#include "pfa.hpp"
int main() {
  double av[10] = { 5, 6, 7, 8, 5, 5, 5, 5, 5, 5 };
  PFA model(av, 5, 2);
  model.Print();
  return 0;
}
