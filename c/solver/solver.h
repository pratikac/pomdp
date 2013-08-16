#ifndef __solver_h__
#define __solver_h__

#include "utils.h"
#include "kdtree.h"

#include "model.h"
#include "belief_tree.h"
#include "simulator.h"

#include "policy.h"
#include "bounds.h"

class solver_t{
  public:
    typedef struct kdtree kdtree_t;
    typedef struct kdres kdres_t;

    model_t* model;
    belief_tree_t* belief_tree;
    kdtree_t* feature_tree;
    bounds_t bounds;

    float insert_distance;
    float convergence_threshold;
    int updates_per_iter;

    solver_t() : model(nullptr), belief_tree(nullptr), feature_tree(nullptr),
              updates_per_iter(1){}
  
    int initialise(belief_t& b_root, model_t* model_in,
        float insert_distance_in=0.01, float convergence_threshold_in=0.01)
    {
      belief_tree = new belief_tree_t(b_root);
      model = model_in;
      feature_tree = kd_create(model->ns);
      insert_into_feature_tree(belief_tree->root);

      insert_distance = insert_distance_in;
      convergence_threshold = convergence_threshold_in;

      bounds.model = model;
      bounds.belief_tree = belief_tree;
      bounds.initialize();
      bounds.calculate(belief_tree->root);

      return 0;
    }
    
    virtual ~solver_t()
    {
      if(feature_tree)
        kd_free(feature_tree);
      if(belief_tree)
        delete belief_tree;
    }

    virtual double* get_key(belief_node_t* bn)
    {
      return bn->b.p.data();
    }

    int insert_into_feature_tree(belief_node_t* bn)
    {
      double* key = get_key(bn);
      kd_insert(feature_tree, key, bn);
      return 0;
    }

    int insert_into_belief_tree(belief_node_t* par, edge_t* e)
    {
      belief_tree->insert(par, e);
      insert_into_feature_tree(e->end);
      return 0;
    }

    bool check_insert_into_belief_tree(belief_node_t* par, edge_t* e)
    {
      belief_node_t* bn = e->end;
      kdres_t* kdres = kd_nearest(feature_tree, get_key(bn));
      bool to_insert = false;
      if(kd_res_end(kdres))
        to_insert = true;
      else
      {
        belief_node_t* bnt = (belief_node_t*)kd_res_item_data(kdres);
        if(bnt != bn)
        {
          float tmp = bnt->b.distance(bn->b);
          //cout<<"tmp: "<< tmp << endl;
          if(tmp > insert_distance)
            to_insert = true;
          else
            to_insert = false;
        }
      }
      kd_res_free(kdres);
      return to_insert;
    }
    
    virtual void sample_child_aid_oid(belief_node_t* par, int& aid, int& oid)
    {
      aid = model->na*RANDF;
      oid = model->no*RANDF;
    }

    virtual edge_t* sample_child_belief(belief_node_t* par)
    {
      int aid, oid;
      sample_child_aid_oid(par, aid, oid);
      belief_t b = model->next_belief(par->b, aid, oid);
      belief_node_t* bn = new belief_node_t(b, par);
      bn->depth = par->depth + 1;

      bounds.calculate(bn);

      edge_t* toret = new edge_t(bn, aid, oid);
      if(!check_insert_into_belief_tree(bn, toret))
      {
        delete toret;
        return NULL;
      }
      return toret;
    }
    
    virtual int sample_belief_nodes()
    {
      vector<pair<belief_node_t*, edge_t*> > nodes_to_insert;
      for(auto& bn : belief_tree->nodes)
      {
        edge_t* e_bn = sample_child_belief(bn);
        if(e_bn)
          nodes_to_insert.push_back(make_pair(bn, e_bn));
      }
      for(auto& p : nodes_to_insert)
        insert_into_belief_tree(p.first, p.second);
      return nodes_to_insert.size();
    }
    
    void prune_belief_tree(belief_node_t* bn)
    {
      int na = model->na;
      vector<float> Qu(na, FLT_MAX/2), Ql(na, -FLT_MAX/2);
      int dummy_oid_opt;

      for(auto& e : bn->children)
      {
        int a = e->aid;
        Qu[a] = bounds.calculate_Q_upper_bound(bn->b, a, dummy_oid_opt);
        Ql[a] = bounds.calculate_Q_lower_bound(bn->b, a, dummy_oid_opt);
      }
      vector<int> toremove(na, 0);
      for(int a2 : range(0,na))
      {
        if((toremove[a2] == 0) && (Qu[a2] < FLT_MAX/2))
        {
          for(int a1 : range(0,na))
          {
            if(a1 != a2)
            {
              if(Ql[a1] > -FLT_MAX/2)
              {
                if((Qu[a2] < Ql[a1]))
                {
                  toremove[a2] = 1;
                  break;
                }
              }
            }
          }
        }
      }
      set<edge_t*> children_copy = bn->children;
      for(auto& e : children_copy)
      {
        int a = e->aid;
        if(toremove[a])
        {
          belief_tree->prune_below(e->end);
        }
      }
      
      for(auto& e : bn->children)
      {
        prune_belief_tree(e->end);
      }
    }
    
    void prune_beliefs()
    {
      //int nbn = belief_tree->nodes.size();
      if(feature_tree)
        kd_free(feature_tree);
      feature_tree = kd_create(model->ns);
      
      for(auto& ce : belief_tree->root->children)
        prune_belief_tree(ce->end);
      
      for(auto& bn : belief_tree->nodes)
        insert_into_feature_tree(bn);

      //cout<<"num_belief_pruned: "<< nbn - belief_tree->nodes.size() << endl;
    }

    void update_nodes()
    {
      bounds.update_all();
#if 1
      prune_beliefs();
      bounds.update_all();
#endif
    }
    
    void iterate()
    {
      sample_belief_nodes();
      for(int i : range(0, updates_per_iter))
        update_nodes();
    }

    virtual bool is_converged()
    {
      return false;
    }
    

    void print_belief_tree()
    {
      belief_tree->print(belief_tree->root);
    }

    virtual float simulate(int steps)
    {
      simulator_t sim(model, bounds);
      return sim.simulate_trajectory(steps, model->b0);
    }
};

#endif
