#include <iostream>
#include <cstdlib>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "eiquadprog.hpp"

using namespace std;
using namespace Eigen;

typedef VectorXd vec;
typedef MatrixXd mat;

int main()
{
  srand(time(NULL));

  // 1-d states
  vec x(4), t(4), ones(4), bn(3);
  x << 0.1, 0.2, 0.3, 0.15;
  t = vec::Random(4);
  ones << 1, 1, 1, 1; 
  /*
  t = t.array()* t.array();
  t += ones;
  */
  bn << 0.4, 0.1, 0.5;

  mat A = mat::Zero(4,4);
  vec b = vec::Zero(4);
  for(int i=0; i<4; i++){
    float t1 = 0;
    for(int j=0; j<4; j++){
      A(i,j) = exp(t(i)*x(j));

      if(j == 3)
        continue;
      else
        t1 += bn(j)*exp(t(i)*x(j));
    }
    b(i) = t1; 
  }
 
  
  // solve min ||Ay-b||^2_2 wrt, y >0, ones'*y = 1
  mat G = 2*A.transpose()*A;
  mat Gc = G;
  vec g0 = -2*b.transpose()*A;
  
  mat CE = ones;
  vec ce0(1); ce0(0) = -1;
  
  mat CI = mat::Identity(4,4);
  vec ci0 = vec::Zero(4);
  
  vec y(4);
  float val = solve_quadprog(Gc, g0, CE, ce0, CI, ci0, y);

  cout<<"val: "<< val + b.dot(b) << endl;
  cout<<"y: "<< y.transpose() << endl;
  
  return 0;
};

