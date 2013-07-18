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

  pbvi_t pbvi;
  pbvi.initialise(m.b0, &m);
  
  tt timer;
  timer.tic();
  for(int i=0; i<num_samples; i++)
  {
    pbvi.sample_belief_nodes();
    for(int j=0; j< backup_per_sample; j++)
      pbvi.backup_belief_nodes();
    
    pbvi.bellman_update_tree();
    cout<<"i: "<< i << endl;
    //pbvi.print_alpha_vectors();
  }
  cout<<timer.toc()<<"[ms]"<<endl;
  cout<<"reward: "<< pbvi.belief_tree->root->value_lower_bound<<endl;
  pbvi.belief_tree->print(pbvi.belief_tree->root);
  
  vec sim_stats(num_sim);
  for(int i=0; i<num_sim; i++)
    sim_stats(i) = pbvi.simulate(num_steps);
  cout<<"mean: "<< sim_stats.mean() << endl;
  return 0;
}

int test_ipomdp(int argc, char** argv)
{
  //srand(time(NULL));
  int ilist[10] = {10, 50, 100, 150, 200, 250, 300, 350, 400, 450};
  for(int i=0; i<2; i++)
  {
    ipomdp_t<lightdark_t<1,1,1>, pbvi_t> ipomdp;
    ipomdp.create_model(ilist[i],4,4);
    ipomdp.solve_model();
  }
  return 0;
}

int main(int argc, char** argv)
{
  test_solver(argc, argv);
  //test_ipomdp(argc, argv);
  return 0;
}
