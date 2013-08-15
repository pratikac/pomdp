#include "parser/mdp.h"

#include "ipomdp.h"

using namespace std;

int test_solver(int argc, char** argv, solver_t& solver)
{
  int c;
  int num_samples = 25, backup_per_sample = 2;
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

  solver.initialise(m.b0, &m, 1e-10, 0.01);
  solver.updates_per_iter = backup_per_sample;

  tt timer;
  timer.tic();
  for(int i=0; i<num_samples; i++)
  {
    solver.iterate();

    cout<<"i: "<< i <<" ("<< solver.belief_tree->root->value_upper_bound<<","<<
      solver.belief_tree->root->value_lower_bound<<") av: "<< solver.alpha_vectors.size() << " bn: "<< 
      solver.belief_tree->nodes.size()<<endl;
    //solver.print_alpha_vectors();
    //getchar();
  }
  cout<<timer.toc()<<"[ms]"<<endl;
  cout<<"reward: "<< solver.belief_tree->root->value_lower_bound<<endl;
  //solver.belief_tree->print(solver.belief_tree->root);
  
  vec sim_stats(num_sim);
  for(int i=0; i<num_sim; i++)
    sim_stats(i) = solver.simulate(num_steps);
  cout<<"mean: "<< sim_stats.mean() << endl;
  return 0;
}

int test_bpomdp(int argc, char** argv)
{
  typedef bpomdp_t<lightdark_t<1,1,1>, sarsop_t> sarsop_lightdark_t;
  typedef bpomdp_t<singleint_t<1,1,1>, sarsop_t> sarsop_singleint_t;
  typedef bpomdp_t<singleint_t<1,1,1>, pbvi_t> pbvi_singleint_t;
  
  //srand(time(NULL));
  int ilist[10] = {10, 50, 100, 150, 200, 250, 300, 350, 400, 450};
  for(int i=3; i<4; i++)
  {
    sarsop_lightdark_t bpomdp(ilist[i],4,4);
    bpomdp.solve();
  }
  return 0;
}

int test_ipomdp(int argc, char** argv)
{
  typedef ipomdp_t<lightdark_t<1,1,1>, sarsop_t> sarsop_lightdark_t;
  typedef ipomdp_t<lightdark_t<1,1,1>, pbvi_t> pbvi_lightdark_t;
  typedef ipomdp_t<singleint_t<1,1,1>, sarsop_t> sarsop_singleint_t;
  typedef ipomdp_t<singleint_t<1,1,1>, pbvi_t> pbvi_singleint_t;
  
  sarsop_lightdark_t pomdp(25, 4, 4);
  pomdp.solve(15);

  //for(int i : range(0, 5))
    //pomdp.refine(2, 0, 0, 50);

  return 0;
}

int main(int argc, char** argv)
{
  pbvi_t pbvi;
  sarsop_t sarsop;

  //test_solver(argc, argv, sarsop);
  //test_bpomdp(argc, argv);
  test_ipomdp(argc, argv);

  return 0;
}
