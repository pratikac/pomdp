#ifndef __pbvi_h__
#define __pbvi_h__

#include "solver.h"

class pbvi_t : public solver_t{
  public:
    typedef struct kdtree kdtree_t;
    typedef struct kdres kdres_t;

    float insert_distance;
    float convergence_threshold;

    pbvi_t(){}

    int initialise(belief_t& b_root, model_t* model_in, 
        float insert_distance_in=0.01, float convergence_threshold_in=0.01) 
    {
      solver_t::initialise(b_root, model_in);
      insert_distance = insert_distance_in;
      convergence_threshold = convergence_threshold_in;
      return 0;
    }

    bool check_insert_into_belief_tree(belief_node_t* par, edge_t* e)
    {
      belief_node_t* bn = e->end;
      kdres_t* kdres = kd_nearestf(feature_tree, get_key(bn));
      bool to_insert = false;
      if(kd_res_end(kdres))
        to_insert = true;
      else
      {
        belief_node_t* bnt = (belief_node_t*)kd_res_item_data(kdres);
        float tmp = bnt->b.distance(bn->b);
        //cout<<"tmp: "<< tmp << endl;
        if(tmp > insert_distance)
          to_insert = true;
      }
      kd_res_free(kdres);
      return to_insert;
    }
    
    int sample_belief_nodes()
    {
      vector<pair<belief_node_t*, edge_t*> > nodes_to_insert;
      for(auto& bn : belief_tree->nodes)
      {
        edge_t* e_bn = sample_child_belief(bn);
        if(!check_insert_into_belief_tree(bn, e_bn))
          delete e_bn;
        else
          nodes_to_insert.push_back(make_pair(bn, e_bn));
      }
      for(auto& p : nodes_to_insert)
        insert_into_belief_tree(p.first, p.second);
      return nodes_to_insert.size();
    }
};


#endif
