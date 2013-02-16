#include <iostream>
#include "linalg.h"
#include "utils.h"
#include "pomdp.h"
using namespace std;

void test_eigen_sparse()
{
  tt times;
  times.tic();

  for(int k=0; k<100; k++)
  {
    spmat A(1000,1000);
    for(int i=0; i<10000; i++)
    {
      int r = rand()%100;
      int c = rand()%100;
      A.coeffRef(r,c) += 1e-3;
    }
    spvec x(1000);
    for(int i=0; i<500;i++)
    {
      int j = rand()%1000;
      x.insert(j) = (i+j)/100.0;
    }
    spvec b = A*x;
  }
  cout<<times.toc()<< " [s]"<<endl;
}

int main()
{
  Belief b;

  return 0;
}
