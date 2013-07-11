#ifndef __solver_h__
#define __solver_h__

#include "utils.h"
#include "kdtree.h"

#include "model.h"
#include "belief_tree.h"
#include "simulator.h"


#define RANDF   (rand()/(RAND_MAX+1.0))

class solver_t;
class policy_t;

class policy_t{
  public:
    vector<alpha_t> alpha_vectors;

    policy_t(vector<alpha_t*>& av)
    {
      alpha_vectors.clear();
      for(auto& pav : av)
        alpha_vectors.push_back(*pav);
    }
};

class solver_t{
  public:
    typedef struct kdtree kdtree_t;
    typedef struct kdres kdres_t;

    model_t* model;
    belief_tree_t* belief_tree;
    kdtree_t* feature_tree;

    vector<alpha_t*> alpha_vectors;
  
    solver_t(){}

    solver_t(belief_t& b_root, model_t* model_in)
    {
      initialise_solver(b_root, model_in);
    }
    int initialise_solver(belief_t& b_root, model_t* model_in)
    {
      belief_tree = new belief_tree_t(b_root);
      model = model_in;
      feature_tree = kd_create(model->ns);
      insert_into_feature_tree(belief_tree->root);

      initiate_alpha_vector();
      return 0;
    }

    int initiate_alpha_vector()
    {
      // create first alpha vector by blind policy
      float max_val = -FLT_MAX;
      int best_action = -1;
      for(int a=0; a<model->na; a++)
      {
        float t1 = model->get_step_reward(a).minCoeff();
        if(t1 > max_val)
        {
          max_val = t1;
          best_action = a;
        }
      }
      vec tmp = vec::Constant(model->ns, max_val/(1.0 - model->discount));
      alpha_t* first_alpha = new alpha_t(best_action, tmp);
      alpha_vectors.push_back(first_alpha);
      return 0;
    }

    virtual ~solver_t()
    {
      if(feature_tree)
        kd_free(feature_tree);
      if(belief_tree)
        delete belief_tree;
      for(auto& pav : alpha_vectors)
        delete pav;
    }

    virtual float* get_key(belief_node_t* bn)
    {
      return bn->b.p.data();
    }

    int insert_into_feature_tree(belief_node_t* bn)
    {
      float* key = get_key(bn);
      kd_insertf(feature_tree, key, bn);
      return 0;
    }

    int insert_into_belief_tree(belief_node_t* par, edge_t* e)
    {
      belief_tree->insert(par, e);
      insert_into_feature_tree(e->end);
      return 0;
    }

    virtual edge_t* sample_child_belief(belief_node_t* par)
    {
      int aid = model->na*RANDF;
      int oid = model->no*RANDF;
      belief_t b = model->next_belief(par->b, aid, oid);
      belief_node_t* bn = new belief_node_t(b, par);
      return new edge_t(bn, aid, oid);
    }
    
    virtual int sample_belief_nodes()
    {
      vector<pair<belief_node_t*, edge_t*> > nodes_to_insert;
      for(auto& bn : belief_tree->nodes)
      {
        edge_t* e_bn = sample_child_belief(bn);
        nodes_to_insert.push_back(make_pair(bn, e_bn));
      }
      for(auto& p : nodes_to_insert)
        insert_into_belief_tree(p.first, p.second);
      return nodes_to_insert.size();
    }

    int find_greater_alpha(const alpha_t& a1, const alpha_t& a2)
    {
      size_t tmp = 0;
      if(a1.grad.size() != a2.grad.size())
        cout<<a1.grad.size()<<","<<a2.grad.size()<<endl;

      vec gd = a1.grad - a2.grad;
      float t1=0, e=0.01;
      for(auto& bn : belief_tree->nodes)
      {
        t1 = gd.dot(bn->b.p);
        if(t1>e)
          tmp++;
        else if(t1 < e)
          tmp--;
      }
      if(tmp == belief_tree->nodes.size())
        return 1;
      else if(tmp == -belief_tree->nodes.size())
        return -1;
      else
        return 0;
    }

    virtual int insert_alpha(alpha_t* a)
    {
      // 1. don't push if too similar to any alpha vector
      for(auto& pav : alpha_vectors)
      {
        if( (a->grad - pav->grad).norm() < 0.01)
          return 1;
      }
      // 2. prune set
      set<alpha_t*> surviving_vectors;
      int broke_out = 0;
#if 1
      for(auto& pav : alpha_vectors)
      {
        int res = find_greater_alpha(*a, *pav);
        if(res == -1)
        {
          delete a;
          broke_out = 1;
          break;
        }
        else if(res == 1)
          surviving_vectors.insert(a);
        else if(res == 0)
        {
          surviving_vectors.insert(pav);
          surviving_vectors.insert(a);
        }
      }
      if(!broke_out)
        alpha_vectors = vector<alpha_t*>(surviving_vectors.begin(), surviving_vectors.end());
#else
      alpha_vectors.push_back(a);
#endif
      return broke_out;
    }

    virtual int backup(belief_node_t* bn)
    {
      int ns = model->ns;
      int no = model->no;
      int na = model->na;

      alpha_t* alpha_a_o[ns][no];
      for(int a=0; a < na; a++)
      {
        for(int o=0; o< no; o++)
        {
          float max_val = -FLT_MAX;
          belief_t nb = model->next_belief(bn->b, a, o);
          for(auto& pav : alpha_vectors)
          {
            float t1 = pav->get_value(nb);
            if(t1 > max_val)
            {
              max_val = t1;
              alpha_a_o[a][o] = pav;
            }
          }
        }
      }

      alpha_t* new_alpha = new alpha_t();
      float max_val = -FLT_MAX;
      for(int a=0; a< na; a++)
      {
        mat t0(ns, no);
        for(int o=0; o<no; o++)
          t0.col(o) = alpha_a_o[a][o]->grad;

        vec t1 = (model->po[a].transpose() * t0.transpose()). diagonal();
        t1 = model->get_step_reward(a) + model->discount*model->pt[a]*t1;
        alpha_t t2(a, t1);
        float t3 = t2.get_value(bn->b);

        if(t3 > max_val)
        {
          max_val = t3;
          *new_alpha = t2;
        }
      }
      //cout<<"inserted: "<<insert_alpha(new_alpha)<<endl;
      insert_alpha(new_alpha);
      
      bn->value_lower_bound = calculate_belief_value(bn->b);
      return 0;
    }
    
    float calculate_belief_value(belief_t& b)
    {
      float max_val = -FLT_MAX;
      for(auto& av : alpha_vectors)
      {
        float t1 = av->get_value(b);
        if(t1 > max_val)
          max_val = t1;
      }
      return max_val;
    }

    virtual int backup_belief_nodes()
    {
      for(auto& bn : belief_tree->nodes)
        backup(bn);
      return 0;
    }

    void print_alpha_vectors()
    {
      for(auto& av : alpha_vectors)
        av->print();
    }

    virtual float simulate(int steps)
    {
      simulator_t sim(model, alpha_vectors);
      return sim.simulate_trajectory(steps, model->b0);
    }
};

#endif
