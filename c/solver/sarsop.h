#ifndef __sarsop_h__
#define __sarsop_h__

#include "solver/solver.h"

class sarsop_t : public pbvi_t{
  public:

    sarsop_t(){}

    int get_best_aid(belief_t& b)
    {
      float max_val = -FLT_MAX;
      int best_aid = -1;
      for(int a=0; a< model->na; a++)
      {
#if 0
        belief_t t2 = model->next_belief(b,a,-1);
        float t1 = projected_belief_upper_bound(t2);
#else
        float t1 = model->next_belief(b,a,-1).p.dot(simplex_upper_bound);
#endif
        if(t1 > max_val)
        {
          max_val = t1;
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
        float t3 = t2.p.dot(simplex_upper_bound) - calculate_belief_value(t2);
        float t4 = t1*(t3 - 0.1*pow(model->discount, bn->depth+1));
        if(t4 > max_val)
        {
          max_val = t4;
          best_oid = o;
        }
      }
      return best_oid;
    }

    edge_t* sample_child_belief(belief_node_t* par)
    {
      int aid = get_best_aid(par->b);
      int oid = get_best_oid(par, aid); 

      belief_t b = model->next_belief(par->b, aid, oid);
      belief_node_t* bn = new belief_node_t(b, par);
      bn->depth = par->depth + 1;
      
      bn->value_upper_bound = bn->b.p.dot(simplex_upper_bound);
      bn->value_lower_bound = calculate_belief_value(bn->b);
      
      return new edge_t(bn, aid, oid);
    }
    
    bool is_converged()
    {
      return fabs((belief_tree->root->value_upper_bound - 
          belief_tree->root->value_lower_bound)/belief_tree->root->value_lower_bound) < convergence_threshold;
    }
};

#endif
