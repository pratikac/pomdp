#ifndef __sarsop_h__
#define __sarsop_h__

#include "solver/solver.h"

class sarsop_t : public solver_t{
  public:

    sarsop_t(){}

    int get_best_aid(belief_t& b)
    {
      float max_val = -FLT_MAX;
      int best_aid = -1;
      int dummy_oid_opt;
      for(int a=0; a< model->na; a++)
      {
        float t4 = bounds.calculate_Q_upper_bound(b, a, dummy_oid_opt);
        if(t4 > max_val)
        {
          max_val = t4;
          best_aid = a;
        }
      }
      return best_aid;
    }

    int get_best_oid(belief_node_t* bn, int aid)
    {
      belief_t& b = bn->b;
      int best_oid = -1;
      float max_val = -FLT_MAX;
      for(int o=0; o< model->no; o++)
      {
        float t1 = model->get_p_o_given_b(b, aid, o);
        belief_t t2 = model->next_belief(b,aid,o);
        float t3 = bounds.calculate_upper_bound(t2) - bounds.calculate_lower_bound(t2);
        float t4 = t1*(t3 - convergence_threshold*pow(model->discount, bn->depth+1));
        if(t4 > max_val)
        {
          max_val = t4;
          best_oid = o;
        }
      }
      return best_oid;
    }
    
    void sample_child_aid_oid(belief_node_t* par, int& aid, int& oid)
    {
      aid = get_best_aid(par->b);
      oid = get_best_oid(par, aid); 
    }

    bool is_converged()
    {
      return false;

      return fabs((belief_tree->root->value_upper_bound - 
          belief_tree->root->value_lower_bound)/belief_tree->root->value_lower_bound) < convergence_threshold;
    }
};

#endif
