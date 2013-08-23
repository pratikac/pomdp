#include "utils.h"

double normal_val(vec& m, mat& v, vec& s)
{
  double t1 = 1.0/pow(2*M_PI, m.rows()/2.0)/sqrt(v.determinant());
  double t2 = exp((-0.5*(m-s).transpose()* v.inverse()*(m-s))[0]);

  return t1*t2;
}

double log_normal_val(vec& m, mat& v, vec& s)
{
  double t1 = log(1.0/pow(2*M_PI, m.rows()/2.0)/sqrt(v.determinant()));
  double t2 = ((-0.5*(m-s).transpose()* v.inverse()*(m-s))[0]);

  return t1 + t2;
}
void log_normalize_vec(vec& v)
{
  vec vc = v;
  double min = v.minCoeff();
  for(int i : range(0,v.rows()))
  {
    double t1 = exp(v(i) + min);
    vc(i) = t1;
  }
  v = vc/vc.sum();
}
