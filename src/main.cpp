// Playground, when in doubt
#include "pfa.hpp"
int main()
{
	double av[10] = { 5, 6, 7, 8, 5, 5, 5, 5, 5, 5 };
	arma::mat m(av, 5, 2, false, true);

	m.print();
	arma::cube c = arma::randu<arma::cube>(2, 3, 2);
	arma::cube s = sum(c);
	c.print();
	s.print();
	return 0;
}


