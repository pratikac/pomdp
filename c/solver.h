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
    vec mdp_value_function;

    solver_t() : model(nullptr), belief_tree(nullptr), feature_tree(nullptr){}

    int initialise(belief_t& b_root, model_t* model_in)
    {
      belief_tree = new belief_tree_t(b_root);
      model = model_in;
      feature_tree = kd_create(model->ns);
      insert_into_feature_tree(belief_tree->root);

      initiate_alpha_vector();
      calculate_mdp_policy();

      belief_tree->root->value_lower_bound = calculate_belief_value(belief_tree->root->b);
      belief_tree->root->value_upper_bound = mdp_value_function.dot(belief_tree->root->b.p);
      
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

    int calculate_mdp_policy()
    {
      int ns = model->ns;
      int na = model->na;
      mdp_value_function = vec::Zero(ns); //mat::Constant(ns,1,-FLT_MAX);
      vec mp = mdp_value_function;
      int c = 0;
      bool is_converged = false;
      while(!is_converged)
      {
        mp = mdp_value_function; 
        mat t1 = mat::Zero(ns,na);
        mat t2 = t1;
        for(int i=0; i<na; i++)
        {
          for(int j=0; j<ns; j++)
            t1(j,i) = model->get_step_reward(j, i);
        }
        for(int i=0; i<na; i++)
          t2.col(i) = t1.col(i) + model->discount*model->pt[i]*mp;
        
        mdp_value_function = t2.rowwise().maxCoeff();
        is_converged = (mp-mdp_value_function).norm() < 100;
        c++;
        if(c > 100)
          cout<<"more than 100 mdp value iterations"<<endl;
      }
      //cout<< "mdp_value: "<< endl << mdp_value_function.transpose() << endl;
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
      bn->depth = par->depth + 1;

      bn->value_upper_bound = bn->b.p.dot(mdp_value_function);
      bn->value_lower_bound = bellman_update(bn);
      
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
        if( (a->grad - pav->grad).norm() < 0.1)
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

        // check this, this looks fishy
        vec t1 = (model->po[a] * (t0.transpose())). diagonal();
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
    
    bool compare_belief_distance(pair<belief_node_t*, 
        float>& p1, pair<belief_node_t*, float>& p2)
    {
      return p1.second < p2.second;
    }
    
    float projected_belief_upper_bound(belief_t& b)
    {
      int ns = model->ns;
      float* key = b.p.data();
      size_t num_nodes = belief_tree->nodes.size();
      float radius = 2.5*pow(log(num_nodes)/(float)num_nodes, 1.0/(float)ns);
      kdres* res = kd_nearest_rangef(feature_tree, key, radius);
      vector< pair<float, belief_node_t*> > dists;
      while(!kd_res_end(res))
      {
        float pos[ns];
        belief_node_t* bnn = (belief_node_t*)kd_res_itemf(res, pos);
        dists.push_back(make_pair((bnn->b.p - b.p).norm(), bnn));
        kd_res_next(res);
      }
      kd_res_free(res);
      sort(dists.begin(), dists.end());
      
      belief_node_t* b1 = dists[0].second;
      belief_node_t* b2 = dists[1].second;
      return (b1->value_upper_bound + b2->value_upper_bound)/2.0;
    }

    virtual int bellman_update(belief_node_t* bn)
    {
      int na = model->na;
      int no = model->no;
      float max_value = -FLT_MAX;
      for(int a=0; a < na; a++)
      {
        float t1 = 0;
        for(int o=0; o< no; o++)
        {
          belief_t nb = model->next_belief(bn->b, a, o);
          t1 += model->get_p_o_given_b(bn->b, a, o)*projected_belief_upper_bound(nb);
        }
        t1 = model->get_expected_step_reward(bn->b, a) + model->discount*t1;
        if(t1 > max_value)
          max_value = t1;
      }
      bn->value_upper_bound = max_value;
      return 0;
    }
    
    virtual int bellman_update_tree(belief_node_t* bn)
    {
      if(bn->children.size() == 0)
        return 0;

      vector<float> t1(model->na,0);
      for(auto& ce : bn->children)
      {
        belief_node_t* cbn = ce->end;
        float t2 = model->get_p_o_given_b(bn->b, ce->aid, ce->oid);
        t1[ce->aid] += t2*cbn->value_upper_bound;
      }
      float max_value = -FLT_MAX;
      for(int a=0; a<model->na; a++)
      {
        if(t1[a] != 0)
        {
          t1[a] += model->get_expected_step_reward(bn->b, a);
          if(t1[a] > max_value)
            max_value = t1[a];
        }
      }
      bn->value_upper_bound = max_value;
      return 0;
    }
    
    virtual int bellman_update_node_tree(belief_node_t* bn)
    {
      for(auto& ce : bn->children)
        bellman_update_node_tree(ce->end);
      bellman_update(bn);
      return 0;
    }
    
    virtual int bellman_update_nodes_tree()
    {
      // do dfs and bellman updates while coming up
      bellman_update_node_tree(belief_tree->root);
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
   
    virtual int bellman_update_nodes()
    {
      for(auto& bn : belief_tree->nodes)
        bellman_update(bn);
      return 0;
    }
    
    virtual int update_nodes()
    {
      backup_belief_nodes();
      bellman_update_nodes_tree();
      return 0;
    }

    virtual bool is_converged(float threshold)
    {
      return false;
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
