#include <iostream>
#include "utils.h"
#include "pbvi.h"
#include "simulator.h"
#include "parser/mdp.h"

using namespace std;

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
  for(int i=0; i<25; i++)
  {
    pbvi.sample_belief_nodes();
    for(int j=0; j< 10; j++)
      pbvi.backup_belief_nodes();

    //pbvi.print_alpha_vectors();
  }
  cout<<timer.toc()<<"[ms]"<<endl;
  cout<<"reward: "<< pbvi.belief_tree->root->value_lower_bound<<endl;
  //pbvi.belief_tree->print(pbvi.belief_tree->root);
  
  vec sim_stats(20);
  for(int i=0; i<20; i++)
    sim_stats(i) = pbvi.simulate(300);
  cout<<"mean: "<< sim_stats.mean() << endl;
  return 0;
}
