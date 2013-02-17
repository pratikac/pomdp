#include <iostream>
#include "linalg.h"
#include "utils.h"
#include "pomdp.h"

using namespace std;
using namespace pomdp;

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

void test1(){
  spmat A(10,10);
  spvec x(10);
  for(int i=0; i<10;i++)
  {
    int r = rand()%10;
    int c = rand()%10;
    A.coeffRef(r,c) += 1e-3;
  }
  for(int i=0; i<5;i++)
  {
    int r = rand()%10;
    x.coeffRef(r) += 100;
  }
  //cout<<A<<endl<<A.col(1).transpose()*x<<endl;
  float* key = vec(x).data();
  for(int i=0; i<10; i++)
    cout<<key[i]<<" ";
  cout<<endl;
}

int main()
{
  /*
  Model m = create_model();    
  //m.print();
  test_model(m);
  */
  
  test1();
  return 0;
}
