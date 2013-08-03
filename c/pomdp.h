#ifndef __pomdp_h__
#define __pomdp_h__

#include "create_model.h"
#include "pbvi.h"
#include "sarsop.h"

typedef struct kdtree kdtree;
typedef struct kdres kdres;

template<class system_t, class solver_t>
class bpomdp_t{
  public:

    create_model_t<system_t> create_model;
    solver_t solver;

    bpomdp_t(int hws, int hwu, int hwo){
      create_model.refine(hws, hwu, hwo);
    }
    ~bpomdp_t(){
    };

    int solve()
    {
      model_t& model = create_model.model;

      solver.initialise(model.b0, &model, 0.1, 0.1);

      for(int i=0; i<1000; i++)
      {
        solver.sample_belief_nodes();
        solver.update_nodes();

        if(i%10 == 0)
          cout<<i << "\t"<<solver.belief_tree->root->value_upper_bound<<"\t"<<solver.belief_tree->root->value_lower_bound << endl;
        //solver.print_alpha_vectors();

        if(solver.is_converged())
          break;
      }
      cout<<"reward: "<< solver.belief_tree->root->value_upper_bound<<" "<<solver.belief_tree->root->value_lower_bound << endl;
      return 0;
    }
};

template<class system_t, class solver_t>
class ipomdp_t{
  public:
    
    model_t* model;
    create_model_t<system_t> create_model;
    solver_t solver;
    
    ipomdp_t(int hws, int hwu, int hwo){
      create_model.refine(hws, hwu, hwo);
      model = &create_model.model;
    }
    ~ipomdp_t(){
    };
    
    void project_belief(belief_node_t& bn)
    {
      int ns = model->ns;
      int ds = create_model.system.ds;
      int nsp = bn.b.p.rows();

      mat A = mat::Zero(ns,ns);
      vec b = vec::Zero(ns);
      vec t = mat::Random(ns,ds).array().abs();
      vec ones = mat::Constant(ns,1,1);

      for(int i=0; i<ns; i++){
        float t1 = 0;
        for(int j=0; j<ns; j++){
          float t2 = exp(t.row(i).dot(create_model.S[j]));
          A(i,j) = t2;

          if(j > nsp-1)
            continue;
          else
            t1 += bn.b.p(j)*t2;
        }
        b(i) = t1; 
      }

      // solve min ||Ay-b||^2_2 wrt, y >0, ones'*y = 1
      mat G = 2*A.transpose()*A;
      mat Gc = G;
      vec g0 = -2*b.transpose()*A;

      mat CE = ones;
      vec ce0(1); ce0(0) = -1;

      mat CI = mat::Identity(ns,ns);
      vec ci0 = vec::Zero(ns);

      vec y(ns);
      float val = solve_quadprog(Gc, g0, CE, ce0, CI, ci0, y);
      
      vec bpp = bn.b.p;
      bn.b.p = y.array().abs();
      bn.b.normalize();
      
      //cout<<"val: "<< val << endl;
      //cout<<"t: "<< t << endl;
      //cout<<"old belief: "<< endl << bpp.transpose() << endl;
      //cout<<"new belief: "<< endl << bn.b.p.transpose() << endl;
      //getchar();
    }

    void project_beliefs()
    {
      kd_free(solver.feature_tree);
      solver.feature_tree = kd_create(model->ns);
      
      for(auto& bn : solver.belief_tree->nodes)
      {
        project_belief(*bn);
        solver.insert_into_feature_tree(bn);
      }
    }

    void project_alpha_vectors()
    {
      int ns = model->ns;
      for(auto& pa : solver.alpha_vectors)
      {
        int nsp = pa->grad.rows();
        vec t1(ns);
        t1.head(nsp) = pa->grad;
        for(int i=nsp; i<ns; i++)
          t1(i) = solver.blind_action_reward; 
        
        pa->grad = t1;
      }
    }
    
    void solver_iteration()
    {
      for(int i=0; i<200; i++)
      {
        solver.sample_belief_nodes();
        solver.update_nodes();

        if(i%10 == 0)
          cout<<i << "\t"<<solver.belief_tree->root->value_upper_bound<<"\t"<<solver.belief_tree->root->value_lower_bound << endl;
        //solver.print_alpha_vectors();

        if(solver.is_converged())
          break;
      }
      cout<<"reward: "<< solver.belief_tree->root->value_upper_bound<<" "<<solver.belief_tree->root->value_lower_bound << endl;
    }

    void refine(int hws, int hwu, int hwo)
    {
      create_model.refine(hws, hwu, hwo);
      project_beliefs();
      project_alpha_vectors();
      
      solver.calculate_mdp_policy();
      solver_iteration();
    }

    void solve()
    {
      solver.initialise(model->b0, model, 0.1, 0.1);
      solver_iteration();
    }

};

#endif