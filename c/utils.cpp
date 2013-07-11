#include "utils.h"

float normal_val(vec& m, mat& v, vec& s)
{
  float t1 = 1.0/pow(2*M_PI, m.rows()/2.0)/sqrt(v.determinant());
  float t2 = exp((-0.5*(m-s).transpose()* v.inverse()*(m-s))[0]);

  return t1*t2;
}
