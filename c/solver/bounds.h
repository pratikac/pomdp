#ifndef __bounds_h__
#define __bounds_h__

#include "model.h"

class bounds_t{
  public:
    model_t* model;
    belief_tree_t* belief_tree;

    vector<alpha_t*> alpha_vectors;
    vec simplex_upper_bound;
    float blind_action_reward;
  
    bounds_t() : blind_action_reward(0) {}
    bounds_t(model_t* min, belief_tree_t* btin) :
      model(min), belief_tree(btin), blind_action_reward(0)
    {
    }
    ~bounds_t()
    {
      for(auto& pav : alpha_vectors)
        delete pav;
    }

    int initialize_blind_policy()
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

    void initialize_upper_bound()
    {
      //calculate_mdp_policy();
      calculate_fib_policy();
    }
    void initialize()
    {
      initialize_blind_policy();
      initialize_upper_bound();
    }

    void calculate(belief_node_t* bn)
    {
      bn->value_lower_bound = calculate_lower_bound(bn->b);
      bn->value_upper_bound = calculate_upper_bound(bn->b);
    }

    int find_greater_alpha(const alpha_t& a1, const alpha_t& a2)
    {
      int max = belief_tree->nodes.size();

      int tmp = 0;
      assert(a1.grad.size() == a2.grad.size());

      vec gd = a1.grad - a2.grad;
      float e=1e-6;
      for(auto& bn : belief_tree->nodes)
      {
        float t1 = gd.dot(bn->b.p);
        if(t1>e)
          tmp++;
        else if(t1 < -e)
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
      float epsilon = 1e-6;
      bool toinsert = true;
      for(auto& pa : alpha_vectors)
      {
        float t1 = (a->grad - pa->grad).array().abs().maxCoeff();
        if(t1< epsilon)
        {
          toinsert = false;
          break;
        }
      }
      if(toinsert)
        alpha_vectors.push_back(a);
      else
        delete a;
      return 0;
    }

    bool pointwise_dominant(alpha_t* a1, alpha_t* a2)
    {
      float epsilon = 1e-6;
      vec t1 = a1->grad - a2->grad;
      float t1p = t1.maxCoeff(), t1m = t1.minCoeff();
      if((t1p*t1m > epsilon) && (t1m >epsilon))
        return true;
      return false;
    }

    virtual int prune_alpha()
    {
      vector<int> dominates(alpha_vectors.size(), 1);
      for(size_t a1=0; a1<alpha_vectors.size(); a1++)
      {
        for(size_t a2=0; a2<alpha_vectors.size(); a2++)
        {
          if((a1 != a2) && (dominates[a2] != 0))
          {
            int res1 = find_greater_alpha(*alpha_vectors[a1], *alpha_vectors[a2]);
            //cout<<"res: "<< res << endl;
            if(res1 == 1)
            {
              dominates[a2] = 0;
            }
          }
        }
      }
      vector<alpha_t*> surviving_vectors;
      for(size_t a1=0; a1<alpha_vectors.size(); a1++)
      {
        if(dominates[a1])
          surviving_vectors.push_back(alpha_vectors[a1]);
        else
          delete alpha_vectors[a1];
      }
      //cout<<"prune_alpha:: were: "<< alpha_vectors.size() << " now: "<< surviving_vectors.size() << endl;
      int num_pruned = alpha_vectors.size() - surviving_vectors.size();
      alpha_vectors = surviving_vectors;
      return num_pruned;
    }

    void backup(belief_node_t* bn)
    {
      int ns = model->ns;
      int no = model->no;
      int na = model->na;

      alpha_t* alpha_a_o[ns][no];
      alpha_t* new_alpha = new alpha_t();
      float max_val2 = -FLT_MAX;
      for(int a=0; a < na; a++)
      {
        mat t0(ns, no);
        for(int o=0; o< no; o++)
        {
          float max_val1 = -FLT_MAX;
          belief_t nb = model->next_belief(bn->b, a, o);
          for(auto& pav : alpha_vectors)
          {
            float t1 = pav->get_value(nb);
            if(t1 > max_val1)
            {
              max_val1 = t1;
              alpha_a_o[a][o] = pav;
            }
          }

          t0.col(o) = alpha_a_o[a][o]->grad;
        }

        // check this, this looks fishy
        vec t2 = (model->po[a] * (t0.transpose())). diagonal();
        vec t3 = model->get_step_reward(a) + model->discount*model->pt[a]*t2;
        float t4 = t3.dot(bn->b.p);
        if(t4 > max_val2)
        {
          max_val2 = t4;
          *new_alpha = alpha_t(a,t2);
        }
      }

      //cout<<"inserted: "<<insert_alpha(new_alpha)<<endl;
      insert_alpha(new_alpha);

      bn->value_lower_bound = calculate_lower_bound(bn->b);
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
      if(belief_tree->nodes.size() == 1)
        return simplex_upper_bound.dot(b.p);
      return sawtooth_project_upper_bound(b);
    }

    float calculate_lower_bound(belief_t& b, alpha_t* alpha_opt=NULL)
    {
      float max_val = -FLT_MAX;
      for(auto& av : alpha_vectors)
      {
        float t1 = av->get_value(b);
        if(t1 > max_val)
        {
          max_val = t1;
          alpha_opt = av;
        }
      }
      return max_val;
    }

    float calculate_Q_bound(belief_t& b, int aid, bool is_lower, int& oid_opt)
    {
      int no = model->no;
      float t1 = 0;

      float t2 = FLT_MAX/2;
      if(is_lower)
        t2 = -FLT_MAX/2;
      oid_opt = -1;
      for(int o : range(0,no))
      {
        belief_t nb = model->next_belief(b, aid, o);
        if(is_lower)
        {
          float t3 = model->get_p_o_given_b(b, aid, o)*calculate_lower_bound(nb);
          t1 += t3;
          if(t3 > t2){
            t2 = t3;
            oid_opt = o;
          }
        }
        else
        {
          float t3 = model->get_p_o_given_b(b, aid, o)*calculate_upper_bound(nb);
          t1 += t3;
          if(t3 < t2){
            t2 = t3;
            oid_opt = o;
          }
        }
      }
      return model->get_expected_step_reward(b, aid) + model->discount*t1;
    }

    float calculate_Q_lower_bound(belief_t& b, int aid, int& oid_opt)
    {
      return calculate_Q_bound(b, aid, true, oid_opt);
    }

    float calculate_Q_upper_bound(belief_t& b, int aid, int& oid_opt)
    {
      return calculate_Q_bound(b, aid, false, oid_opt);
    }
    
    void bellman_update(belief_node_t* bn, int& aid_opt)
    {
      int na = model->na;
      float max_value = -FLT_MAX;
      int dummy_oid_opt;
      aid_opt = -1;
      for(int a=0; a < na; a++)
      {
        float t1 = calculate_Q_upper_bound(bn->b, a, dummy_oid_opt);
        if(t1 > max_value)
        {
          max_value = t1;
          aid_opt = a;
        }
      }
      bn->value_upper_bound = max_value;
    }
    
    void bellman_update_tree(belief_node_t* bn)
    {
      if(bn->children.size() == 0)
        return;

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
    }
    
    void bellman_update_node_tree(belief_node_t* bn)
    {
      int dummy_aid_opt;
      for(auto& ce : bn->children)
        bellman_update_node_tree(ce->end);
      bellman_update(bn, dummy_aid_opt);
    }
    
    void bellman_update_nodes_tree()
    {
      // do dfs and bellman updates while coming up
      bellman_update_node_tree(belief_tree->root);
    }

    void bellman_update_nodes()
    {
      int dummy_aid_opt;
      for(auto& bn : belief_tree->nodes)
        bellman_update(bn, dummy_aid_opt);
    }

    void backup_belief_nodes()
    {
      for(auto& bn : belief_tree->nodes)
        backup(bn);

      //int nalpha = alpha_vectors.size();
      //prune_alpha();
      //cout<<"num_alpha_pruned: "<< nalpha - alpha_vectors.size() << endl;
    }

    void backup_belief_nodes_tree(belief_node_t* bn)
    {
      for(auto& ce : bn->children)
        backup_belief_nodes_tree(ce->end);
      backup(bn);
    }

    void calculate_all()
    {
      for(auto& bn : belief_tree->nodes)
        bn->value_lower_bound = calculate_lower_bound(bn->b);
    }

    void update_all()
    {
      backup_belief_nodes();
      bellman_update_nodes();
    }
    
    void print_alpha_vectors()
    {
      for(auto& av : alpha_vectors)
        av->print();
    }
};

#endif
