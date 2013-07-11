#include "pbvi.h"
#include "simulator.h"
#include "parser/mdp.h"

#include "ipomdp.h"

using namespace std;

int test_solver(int argc, char** argv)
{
  int c;
  int num_samples = 25, backup_per_sample = 5;
  int num_sim = 20, num_steps = 300;
  char* fname = NULL;
  while( (c = getopt(argc, argv, "f:n:b:s:t:")) != -1)
  {
    switch(c)
    {
      case 'f':
        fname = optarg;
        break;
      case 'n':
        num_samples = atoi(optarg);
        break;
      case 'b':
        backup_per_sample = atoi(optarg);
        break;
      case 's':
        num_sim = atoi(optarg);
        break;
      case 't':
        num_steps = atoi(optarg);
        break;
      default:
        abort();
    }
  }
  
  model_t m;
  if(fname)
    readMDP(&(fname[0]), m);
  else
    m = create_example();
  m.print();
  getchar();

  belief_t b0 = m.b0;
  pbvi_t pbvi(b0, &m);
  
  tt timer;
  timer.tic();
  for(int i=0; i<num_samples; i++)
  {
    pbvi.sample_belief_nodes();
    for(int j=0; j< backup_per_sample; j++)
      pbvi.backup_belief_nodes();

    //pbvi.print_alpha_vectors();
  }
  cout<<timer.toc()<<"[ms]"<<endl;
  cout<<"reward: "<< pbvi.belief_tree->root->value_lower_bound<<endl;
  //pbvi.belief_tree->print(pbvi.belief_tree->root);
  
  vec sim_stats(num_sim);
  for(int i=0; i<num_sim; i++)
    sim_stats(i) = pbvi.simulate(num_steps);
  cout<<"mean: "<< sim_stats.mean() << endl;
  return 0;
}

int test_ipomdp(int argc, char** argv)
{
  ipomdp_t<lightdark_t<1,1,1>, pbvi_t> ipomdp;
  ipomdp.create_model();
  ipomdp.model.print("model.pomdp");
  getchar();
  ipomdp.solve_model();
  return 0;
}

int main(int argc, char** argv)
{
  //test_solver(argc, argv);
  test_ipomdp(argc, argv);
  return 0;
}
