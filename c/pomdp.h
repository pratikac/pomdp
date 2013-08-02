#ifndef __pomdp_h__
#define __pomdp_h__

#include "create_model.h"
#include "pbvi.h"
#include "sarsop.h"

typedef struct kdtree kdtree;
typedef struct kdres kdres;

template<class system_t, class solver_t>
class bpomdp_t{
  public:

    create_model_t<system_t> create_model;
    solver_t solver;

    bpomdp_t(int hws, int hwu, int hwo){
      create_model.initialise(hws, hwu, hwo);
    }
    ~bpomdp_t(){
    };

    int solve()
    {
      model_t& model = create_model.model;

      solver.initialise(model.b0, &model, 0.1, 0.1);

      for(int i=0; i<1000; i++)
      {
        solver.sample_belief_nodes();
        solver.update_nodes();

        if(i%10 == 0)
          cout<<i << "\t"<<solver.belief_tree->root->value_upper_bound<<"\t"<<solver.belief_tree->root->value_lower_bound << endl;
        //solver.print_alpha_vectors();

        if(solver.is_converged())
          break;
      }
      cout<<"reward: "<< solver.belief_tree->root->value_upper_bound<<" "<<solver.belief_tree->root->value_lower_bound << endl;
      return 0;
    }
};

template<class system_t, class solver_t>
class ipomdp_t{
  public:
     
    create_model_t<system_t> create_model;
    solver_t solver;
    
    ipomdp_t(int hws, int hwu, int hwo){
      create_model.initialise(hws, hwu, hwo);
    }
    ~ipomdp_t(){
    };
   
    int solve()
    {
      model_t& model = create_model.model;

      solver.initialise(model.b0, &model, 0.1, 0.1);

      for(int i=0; i<1000; i++)
      {
        solver.sample_belief_nodes();
        solver.update_nodes();
        
        if(i%10 == 0)
          cout<<i << "\t"<<solver.belief_tree->root->value_upper_bound<<"\t"<<solver.belief_tree->root->value_lower_bound << endl;
        //solver.print_alpha_vectors();
        
        if(solver.is_converged())
          break;
      }
      cout<<"reward: "<< solver.belief_tree->root->value_upper_bound<<" "<<solver.belief_tree->root->value_lower_bound << endl;
      return 0;
    }
};

#endif
