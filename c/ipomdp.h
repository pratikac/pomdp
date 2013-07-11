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
      r = 2.5*pow(log(ns)/(float)ns, 1.0/(float)ns);
      gamma = 0.99;
      epsilon = 1e-15;

      tree = kd_create(ns);
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
        kd_insertf(tree, key.data(), i);
      }
      r = 2.5*pow(log(ns)/(float)ns, 1.0/(float)ns);
      return 0;
    }
    int sample_controls()
    {
      for(int i=num; i<nu; i++)
        U.push_back(system.sample_control());
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
      return 0;
    }
    int sample_all()
    {
      sample_states();
      sample_controls();
      sample_observations();

      get_holding_time();
      
      get_b0();

      return 0;
    }
    
    int get_P()
    {
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

            vec probs = Zero(ns);
            float pos[ds] = {0};
            while(!kd_res_end(res))
            {
              int* skindex = (int*)kd_res_itemf(res,pos);
              vec& sk = S[*skindex];
              if(!system.is_in_obstacle(s,sk))
                probs(*skindex) = normal_val(s, FFdt, sk);
              
              kd_res_next(res);
            }
            model.pt_t[j].row(i) = probs.transpose();
            kd_res_rewind(res);
          }
        }
        kd_res_free(res);
      }
      return 0;
    }

    int get_Q()
    {
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

      return 0;
    }

    int create_model()
    {
      get_P();
      get_Q();
      get_R();
      
      model.ns = ns;
      model.na = nu;
      model.no = no;
      model.discount = gamma;
      model.b0 = get_b0();
      model.normalize_mat();

      return 0;
    }
};

#endif
