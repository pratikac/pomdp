#include <iostream>
#include "utils.h"
#include "pbvi.h"
#include "simulator.h"
#include "parser/mdp.h"

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
  /*
     float* key = vec(x).data();
     for(int i=0; i<10; i++)
     cout<<key[i]<<" ";
     cout<<endl;
     */
  mat A1 = mat::Identity(10,10);
  vec Ad = A1.diagonal();

  vec v1 = vec::Random(10);
  //cout<<"entropy: "<< v1.dot(v1.array().sin().matrix()) <<endl;
}

int test2()
{
  vector<int*> pvec;
  for(int i=0; i<10; i++)
  {
    int* t = new int;
    *t = i;
    pvec.push_back(t);
  }
  for(auto i : pvec)
  {
    cout<< *i<<" ";
    delete i;
  }
  cout<<endl;
  return 0;
}

int main(int argc, char** argv)
{
  
  string fname = "";
  model_t m;
  if(argc > 1)
    fname = argv[1];
  else
  {
    cout<<"no input file or model"<<endl;
    exit(0);
  }
  if(fname != "")
    readMDP(&(fname[0]), m);
  else
    m = create_example();
  m.print();
  getchar();

  belief_t b0 = m.b0;
  pbvi_t pbvi(b0, &m);
  
  tt timer;
  timer.tic();
  for(int i=0; i<10; i++)
  {
    pbvi.sample_belief_nodes();
    for(int j=0; j< 1; j++)
      pbvi.backup_belief_nodes();

    //pbvi.print_alpha_vectors();
    cout<<i<<endl;
  }
  cout<<timer.toc()<<"[ms]"<<endl;
  cout<<"reward: "<< pbvi.belief_tree->root->value_lower_bound<<endl;
  //pbvi.belief_tree->print(pbvi.belief_tree->root);
  
  /*
  vec sim_stats(20);
  for(int i=0; i<20; i++)
    sim_stats(i) = pbvi.simulate(300);
  cout<<"mean: "<< sim_stats.mean() << endl;
  */
  return 0;
}
