#ifndef __solver_h__
#define __solver_h__

#include "utils.h"
#include "kdtree.h"

#include "model.h"
#include "belief_tree.h"
#include "simulator.h"

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
    float insert_distance;
    float convergence_threshold;
    
    float blind_action_reward;
    vector<alpha_t*> alpha_vectors;
    vec simplex_upper_bound;

    solver_t() : model(nullptr), belief_tree(nullptr), feature_tree(nullptr),
              blind_action_reward(0){}
  
    int initialise(belief_t& b_root, model_t* model_in,
        float insert_distance_in=0.01, float convergence_threshold_in=0.01)
    {
      belief_tree = new belief_tree_t(b_root);
      model = model_in;
      feature_tree = kd_create(model->ns);
      insert_into_feature_tree(belief_tree->root);

      insert_distance = insert_distance_in;
      convergence_threshold = convergence_threshold_in;

      initialise_blind_policy();
      calculate_initial_upper_bound();

      belief_tree->root->value_lower_bound = calculate_lower_bound(belief_tree->root->b);
      belief_tree->root->value_upper_bound = simplex_upper_bound.dot(belief_tree->root->b.p);
      
      return 0;
    }
    
    int initialise_blind_policy()
    {
      int ns = model->ns;
      int na = model->na;
      float t1 = FLT_MAX;
      for(int a=0; a<na; a++)
      {
        float t2 = model->get_step_reward(a).minCoeff();
        if(t2 < t1)
          t1 = t2;
      }
      blind_action_reward = t1/(1-model->discount);
      mat Qlas = mat::Constant(ns, na, blind_action_reward);
      
      mat Qlasc = Qlas;
      bool is_converged = false;
      while(!is_converged){

        for(int a=0; a<na; a++)
          Qlas.col(a) = model->get_step_reward(a) + model->discount*model->pt[a]*Qlasc.col(a);
        is_converged = (Qlas-Qlasc).array().abs().maxCoeff() < 0.1;
        Qlasc = Qlas;
      }
      for(int a=0; a<na; a++)
      {
        vec t3 = Qlas.col(a);
        alpha_t* alpha = new alpha_t(a, t3);
        insert_alpha(alpha);
      }
      //cout<<"blind policy initialized"<<endl;
      return 0;
    }
    
    
    int calculate_mdp_policy()
    {
      int ns = model->ns;
      int na = model->na;
      simplex_upper_bound = vec::Constant(ns, 0);
      vec mp = simplex_upper_bound;
      int c = 0;
      bool is_converged = false;
      while(!is_converged)
      {
        mp = simplex_upper_bound; 
        mat t1 = mat::Zero(ns,na);
        mat t2 = t1;
        for(int i=0; i<na; i++)
        {
          for(int j=0; j<ns; j++)
            t1(j,i) = model->get_step_reward(j, i);
        }
        for(int i=0; i<na; i++)
          t2.col(i) = t1.col(i) + model->discount*model->pt[i]*mp;

        simplex_upper_bound = t2.rowwise().maxCoeff();
        is_converged = (mp-simplex_upper_bound).norm() < 0.1;
        c++;
        if(c > 1000)
        {
          cout<<"more than 1000 mdp value iterations"<<endl;
          //cout<< "mdp_value: "<< endl << simplex_upper_bound.transpose() << endl;
        }
      }
      //cout<< "mdp_value: "<< endl << simplex_upper_bound.transpose() << endl;
      return 0;
    }
    
    int calculate_fib_policy()
    {
      int ns = model->ns;
      int na = model->na;
      int no = model->no;

      float t1 = -FLT_MAX/2;
      for(int a=0; a<na; a++)
      {
        float t2 = model->get_step_reward(a).maxCoeff();
        if(t2 > t1)
          t1 = t2;
      }
      mat Qas = mat::Constant(ns, na, t1/(1-model->discount));
      
      mat Qasc = Qas;
      bool is_converged = false;
      while(!is_converged){
        for(int a=0; a<na; a++)
        {
          Qas.col(a) = model->get_step_reward(a);
          for(int o1=0; o1<no; o1++)
          {
            vec t3 = model->get_p_a_o(a,o1)*Qasc.rowwise().maxCoeff();
            Qas.col(a) += model->discount*t3;
          }
        }
        is_converged = (Qasc-Qas).norm() < 0.1;
        Qasc = Qas;
      }
      //cout<<"fib converged"<<endl;
      simplex_upper_bound = Qas.rowwise().maxCoeff();
      return 0;
    }
    
    void calculate_initial_upper_bound()
    {
      //calculate_mdp_policy();
      calculate_fib_policy();
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

    virtual edge_t* sample_child_belief(belief_node_t* par)
    {
      int aid = model->na*RANDF;
      int oid = model->no*RANDF;
      
      belief_t b = model->next_belief(par->b, aid, oid);
      belief_node_t* bn = new belief_node_t(b, par);
      bn->depth = par->depth + 1;

      bn->value_upper_bound = calculate_upper_bound(bn->b);
      bn->value_lower_bound = calculate_lower_bound(bn->b);
      
      return new edge_t(bn, aid, oid);
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
    
    virtual void prune_belief_tree()
    {
    }
    
    int find_greater_alpha(const alpha_t& a1, const alpha_t& a2)
    {
      int max = belief_tree->nodes.size();
      if(max < 50)
        return 0;

      int tmp = 0;
      assert(a1.grad.size() == a2.grad.size());

      vec gd = a1.grad - a2.grad;
      float t1=0, e=0;
      for(auto& bn : belief_tree->nodes)
      {
        t1 = gd.dot(bn->b.p);
        if(t1>e)
          tmp++;
        else
          tmp--;
      }
      if(tmp == max)
        return 1;
      else if(tmp == -max)
        return -1;
      else
        return 0;
    }

    virtual int insert_alpha(alpha_t* a)
    {
#if 0
      float epsilon = 1e-10;
      for(auto& pav : alpha_vectors)
      {
        vec t1 = (a->grad-pav->grad);
        float t1p = t1.maxCoeff(), t1m = t1.minCoeff();
        if((t1p <= -t1m) && (t1p < epsilon))
        {
          delete a;
          return 1;
        }
      }
#else
      alpha_vectors.push_back(a);
      return 0;
#endif
    }
    
    virtual int prune_alpha()
    {
      float epsilon = 1e-10;
      vector<int> not_dominated(alpha_vectors.size(), 1);

      for(size_t a1=0; a1<alpha_vectors.size(); a1++)
      {
        for(size_t a2=0; a2<alpha_vectors.size(); a2++)
        {
          if(a1 != a2)
          {
            if(not_dominated[a2])
            {
#if 0
              int res = find_greater_alpha(*alpha_vectors[a1], *alpha_vectors[a2]);
              if( (res == 1) || (res==0) ){
                not_dominated[a1] = false;
                break;
              }
#else
              vec t1 = alpha_vectors[a1]->grad - alpha_vectors[a2]->grad;
              float t1p = t1.maxCoeff(), t1m = t1.minCoeff();
              if((t1p <= -t1m) && (t1p < epsilon)){
                not_dominated[a1] = 0;
                break;
              }
#endif
            }
          }
        }
      }
      vector<alpha_t*> surviving_vectors;
      for(size_t a1=0; a1<alpha_vectors.size(); a1++)
      {
        if(not_dominated[a1])
          surviving_vectors.push_back(alpha_vectors[a1]);
        else
          delete alpha_vectors[a1];
      }
      //cout<<"prune_alpha:: were: "<< alpha_vectors.size() << " now: "<< surviving_vectors.size() << endl;
      alpha_vectors = surviving_vectors;
      return 0;
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
        {
          vec t5 = alpha_a_o[a][o]->grad;
          t0.col(o) = t5;
        }
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
      
      bn->value_lower_bound = calculate_lower_bound(bn->b);
      return 0;
    }
    
    float sawtooth_project_upper_bound(belief_t& b)
    {
      int ns = model->ns;
      float v = simplex_upper_bound.dot(b.p);
      float t2 = FLT_MAX/2;
      for(auto& bn : belief_tree->nodes)
      {
        float t1 = simplex_upper_bound.dot(bn->b.p);
        float t3 = FLT_MAX/2;
        for(int i=0; i<ns; i++)
        {
          float t4 = b.p(i)/bn->b.p(i);
          if(t3 > t4)
            t3 = t4;
        }
        t1 = v + (bn->value_upper_bound - t1)*t3;
        if(t1 < t2)
          t2 = t1;
      }
      return t2;
    }
    
    float projected_belief_upper_bound(belief_t& b)
    {
      int ns = model->ns;
      vec& bp = b.p;
      float t1 = simplex_upper_bound.dot(bp);
      float min_val = FLT_MAX/2;
      for(auto& bn : belief_tree->nodes)
      {
        vec& bnbp = bn->b.p;
        float t4 = -FLT_MAX/2;
        float t3 = bp.dot(bnbp)*bn->value_upper_bound;
        for(int s=0; s<ns; s++){
          float t2 = t1 - simplex_upper_bound(s)*bp(s);
          t2 += t3;
          if(t2 > t4)
            t4 = t2;
        }
        if(t4 < min_val)
          min_val = t4;
      }
      return min_val;
    }
    
    float calculate_upper_bound(belief_t & b)
    {
      return sawtooth_project_upper_bound(b);
    }

    float calculate_lower_bound(belief_t& b)
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
    
    float calculate_Q_bound(belief_t& b, int aid, bool is_lower)
    {
      int no = model->no;
      float t1 = 0; 
      for(int o : range(0,no))
      {
        belief_t nb = model->next_belief(b, aid, o);
        if(is_lower)
          t1 += model->get_p_o_given_b(b, aid, o)*calculate_lower_bound(nb);
        else
          t1 += model->get_p_o_given_b(b, aid, o)*calculate_upper_bound(nb);
      }
      return model->get_expected_step_reward(b, aid) + model->discount*t1;
    }

    float calculate_Q_lower_bound(belief_t& b, int aid)
    {
      return calculate_Q_bound(b, aid, true);
    }

    float calculate_Q_upper_bound(belief_t& b, int aid)
    {
      return calculate_Q_bound(b, aid, false);
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
          t1 += model->get_p_o_given_b(bn->b, a, o)*calculate_upper_bound(nb);
          //t1 += model->get_p_o_given_b(bn->b, a, o)*projected_belief_upper_bound(nb);
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

    virtual int backup_belief_nodes()
    {
      for(auto& bn : belief_tree->nodes)
        backup(bn);
      prune_alpha();
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
      bellman_update_nodes();
      prune_belief_tree();
      return 0;
    }

    void update_bounds()
    {
      for(auto& bn : belief_tree->nodes)
        bn->value_lower_bound = calculate_lower_bound(bn->b);
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
