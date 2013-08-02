#ifndef __create_model_h__
#define __create_model_h__

#include "model.h"
#include "solver.h"
#include "system/lightdark.h"

template<class system_t>
class create_model_t{
  public:

    int ns, nu, no;
    int ds, du, ddo;
    
    model_t model;
    system_t system;
    
    vector<vec> S;
    vector<vec> U;
    vector<vec> O;
    vector<int> linear;
    
    float ht, r, gamma, epsilon;

    kdtree* tree;
    
    create_model_t(){
      ds = system.ds;
      du = system.du;
      ddo = system.ddo;
      
      ns = 0;
      nu = 0; //2.0*log(ns);
      no = 0;
      
      ht = 0.1;
      r = 2.5*pow(log(ns)/(float)ns, 1.0/(float)ds);
      gamma = 0.99;
      epsilon = 1e-15;

      // linear stores [0, 1, 2, .... n-1] (for kdtree hack)
      linear = vector<int>(1000,0);
      for(int i=0; i<1000; i++)
        linear[i] = i;
    }

    ~create_model_t(){
      kd_free(tree);
    }
    
    int sample_states()
    {
      for(int i=0; i<ns; i++)
      {
        vec s = system.sample_state();
        S.push_back(s);
        vec t1 = system.get_key(s);
        double* key = t1.data();
        kd_insert(tree, key, &linear[i]);
        //cout<<"inserted: "<< s <<" -- " << key << " -- " << linear[i] << endl;
      }
      r = 2.5*pow(log(ns)/(float)ns, 1.0/(float)ds);
      return 0;
    }
    
    int sample_controls()
    {
      for(int i=0; i<nu-1; i++)
        U.push_back(system.sample_control());
      U.push_back(system.zero_control());
      return 0;
    }
    int sample_observations()
    {
      for(int i=0; i<no; i++)
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
   
    int test_kd_tree()
    {
      for(int i=0; i<ns; i++)
      {
        cout<<i<< " -- "<<S[i]<<" -- "<<system.get_key(S[i])<< endl;
        vec keyv = system.get_key(S[i]);
        float* key = keyv.data();
        kdres* res = kd_nearest_rangef(tree, key, r);
        while(!kd_res_end(res))
        {
          float pos[ds];
          int* index = (int*)kd_res_itemf(res, pos);
          cout<<"\t: "<< S[*index] << endl;
          kd_res_next(res);
        }
        kd_res_free(res);
      }
      getchar();
      return 0;
    }
    
    int get_P()
    {
      model.pt = vector<mat>(nu, mat::Zero(ns,ns));

      for(int i=0; i<ns; i++)
      {
        vec& s = S[i];
        vec keyv = system.get_key(s);
        double* key = keyv.data();
        kdres* res = kd_nearest_range(tree, key, r);
        if(kd_res_size(res) == 0)
        {
          //cout<< s.transpose()<< " - "<< system.get_key(s).transpose()<< " -- " << r << endl;
          continue;
        }
        else
        {
          for(int j=0; j<nu; j++)
          {
            vec& u = U[j];
            vec sfdt = s + system.get_fdt(s,u,ht);
            mat FFdt = system.get_FFdt(s,u,ht);

            vec probs = vec::Zero(ns);
            double pos[ds];
            while(!kd_res_end(res))
            {
              int* skindex = (int*)kd_res_item(res, pos);
              vec& sk = S[*skindex];
              if(!system.is_in_obstacle(s,sk))
                probs(*skindex) = normal_val(sfdt, FFdt, sk);
              
              kd_res_next(res);
            }
            model.pt[j].row(i) = probs.transpose();
            //cout<<i<<" "<<j<< endl << probs.transpose() << endl;
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
          {
            float t1 = system.get_reward(S[j], U[i], S[k], ht);
            model.pr[i](j,k) = t1;
          }
        }
      }
      return 0;
    }

    int initialise(int ns_, int nu_, int no_)
    {
      ns = ns_;
      nu = nu_;
      no = no_;

      tree = kd_create(ds);

      sample_all();
      get_P();
      get_Q();
      get_R();
      
      model.ns = ns;
      model.na = nu;
      model.no = no;
      model.discount = exp(-0.1*ht);
      cout<<"ht, model.discount: "<< ht<<" "<<model.discount << endl;
      model.b0.p = get_b0();
      model.normalize_mat();

      return 0;
    }

    int print()
    {
      ofstream fout;
      fout.open("model.pomdp");

      fout<<"states: "<<ns<<endl;
      for(int i=0; i<ns; i++)
        fout<< i<< " -- " << S[i].transpose() << endl;
      fout<<"actions: "<<nu<<endl;
      for(int i=0; i<nu; i++)
        fout<< i<< " -- " << U[i].transpose() << endl;
      fout<<"observations: "<< no << endl;
      for(int i=0; i<no; i++)
        fout<< i<< " -- " << O[i].transpose() << endl;
      fout<<"transition_probabilities: "<<endl;
      for(int i=0; i<nu; i++)
      {
        fout<<"aid: "<<i<< endl << model.pt[i] <<endl;
      }
      fout<<"observation_probabilities: "<<endl;
      for(int i=0; i<nu; i++)
      {
        fout<<"aid: "<<i<< endl << model.po[i] <<endl;
      }

      fout<<"initial_belief: "<< model.b0.p.transpose()<<endl;
      fout<<"discount: "<< model.discount <<endl;
      fout<<"reward_function: "<<endl;
      for(int i=0; i<nu; i++)
        fout<<"aid: "<<i<< endl << model.pr[i] <<endl;

      fout.close();
      return 0;
    }
};

#endif

