#ifndef __ipomdp_h__
#define __ipomdp_h__

#include "model.h"
#include "pbvi.h"
#include "system/lightdark.h"

typedef struct kdtree kdtree;
typedef struct kdres kdres;

template<class system_t, class solver_t>
class ipomdp_t{
  public:
    int ns, nsm, nu, num, no, nom;
    int ds, du, ddo;

    system_t system;
    model_t model;
    solver_t solver;
    
    vector<vec> S;
    vector<vec> U;
    vector<vec> O;
    vector<int> linear;
    
    float ht, r, gamma, epsilon;

    kdtree* tree;

    ipomdp_t(){
      ds = system.ds;
      du = system.du;
      ddo = system.ddo;

      nsm = 0;
      ns = 10;
      num = 0;
      nu = 2.0*log(ns);
      nom = 0;
      no = nu;

      ht = 0.1;
      r = 2.5*pow(log(ns)/(float)ns, 1.0/(float)ds);
      gamma = 0.99;
      epsilon = 1e-15;

      tree = kd_create(ns);

      // linear stores [0, 1, 2, .... n-1] (for kdtree hack)
      linear = vector<int>(ns,0);
      for(int i=0; i<ns; i++)
        linear[i] = i;
    }
    ~ipomdp_t()
    {
      kd_free(tree);
    };
   
    int sample_states()
    {
      for(int i=nsm; i<ns; i++)
      {
        vec s = system.sample_state();
        S.push_back(s);
        vec key = system.get_key(s);
        kd_insertf(tree, key.data(), &linear[i]);
      }
      r = 2.5*pow(log(ns)/(float)ns, 1.0/(float)ds);
      return 0;
    }
    int sample_controls()
    {
      for(int i=num; i<nu-1; i++)
        U.push_back(system.sample_control());
      U.push_back(system.zero_control());
      return 0;
    }
    int sample_observations()
    {
      for(int i=nom; i<no; i++)
        O.push_back(system.sample_observation());
      return 0;
    }
    int get_holding_time()
    {
      ht = 1000;
      for(int i=0; i<ns; i++)
      {
        for(int j=0; j< nu; j++)
          ht = min(ht, system.get_ht(S[i], U[j],r));
      }
      ht = 0.99*ht;
      return 0;
    }
    vec get_b0()
    {
      vec t = vec::Zero(ns);
      for(int i=0; i<ns; i++)
        t(i) = normal_val(system.init_state, system.init_var, S[i]);
      return t/t.sum();
    }
    int sample_all()
    {
      sample_states();
      sample_controls();
      sample_observations();

      get_holding_time();
      
      return 0;
    }
    
    int get_P()
    {
      model.pt = vector<mat>(nu, mat::Zero(ns,ns));

      kdres* res = NULL;
      for(int i=0; i<ns; i++)
      {
        vec& s = S[i];

        res = kd_nearest_rangef(tree, system.get_key(s).data(), r);
        if(kd_res_size(res) == 0)
          continue;
        else
        {
          kd_res_rewind(res);
          for(int j=0; j<nu; j++)
          {
            vec& u = U[j];
            vec fdt = system.get_fdt(s,u,ht);
            mat FFdt = system.get_FFdt(s,u,ht);

            vec probs = vec::Zero(ns);
            float pos[ds];
            while(!kd_res_end(res))
            {
              int* skindex = (int*)kd_res_itemf(res,pos);
              vec& sk = S[*skindex];
              if(!system.is_in_obstacle(s,sk))
                probs(*skindex) = normal_val(s, FFdt, sk);
              
              kd_res_next(res);
            }
            model.pt[j].row(i) = probs.transpose();
            kd_res_rewind(res);
          }
        }
        kd_res_free(res);
      }
      return 0;
    }

    int get_Q()
    {
      model.po = vector<mat>(nu, mat::Zero(ns,no));
      mat Q = mat::Zero(ns,no);
      for(int i=0; i<ns; i++)
      {
        for(int j=0; j<no; j++)
          Q(i,j) = system.get_observation_prob(S[i], O[j]);
      }
      for(int i=0; i<nu; i++)
        model.po[i] = Q;
      
      return 0;
    }
    
    int get_R()
    {
      model.pr = vector<mat>(nu, mat::Zero(ns,ns));
      for(int i=0; i<nu; i++)
      {
        for(int j=0; j<ns; j++)
        {
          for(int k=0; k<ns; k++)
            model.pr[i](j,k) = system.get_reward(S[j], U[i], S[k]);
        }
      }
      return 0;
    }

    int create_model()
    {
      sample_all();
      
      get_P();
      get_Q();
      get_R();
      
      model.ns = ns;
      model.na = nu;
      model.no = no;
      model.discount = gamma;
      model.b0.p = get_b0();
      model.normalize_mat();

      return 0;
    }
    
    int solve_model()
    {
      solver = solver_t(model.b0, &model);

      for(int i=0; i<20; i++)
      {
        solver.sample_belief_nodes();
        for(int j=0; j<5; j++)
          solver.backup_belief_nodes();
      }
      cout<<"reward: "<< solver.belief_tree->root->value_lower_bound << endl;
      return 0;
    }
};

#endif
