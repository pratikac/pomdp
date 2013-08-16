#ifndef __ipomdp_h__
#define __ipomdp_h__

#include "create_model.h"
#include "pbvi.h"
#include "sarsop.h"
#include "quadprog.h"

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

      solver.initialise(model.b0, &model);

      for(int i=0; i<1000; i++)
      {
        solver.iterate();

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

#if 0
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
      solve_quadprog(Gc, g0, CE, ce0, CI, ci0, y);
      
      if( (((y-y).array() != (y-y).array()).any()) ||
          (y.sum() < 0.5))
      {
        y = bn.b.p;
        for(int i : range(nsp, ns))
          y(i) = 0;
      }
      vec bpp = bn.b.p;
      bn.b.p = y.array().abs();
      bn.b.normalize();
      
      //cout<<"val: "<< val << endl;
      //cout<<"t: "<< t << endl;
      //cout<<"old belief: "<< endl << bpp.transpose() << endl;
      //cout<<"new belief: "<< endl << bn.b.p.transpose() << endl;
      //getchar();
#else
      vec t1 = vec::Zero(ns);
      t1.head(nsp) = bn.b.p;
      bn.b.p  = t1;
#endif
    }

    void project_beliefs()
    {
      if(solver.feature_tree)
        kd_free(solver.feature_tree);
      solver.feature_tree = kd_create(model->ns);
    
      solver.belief_tree->root->b = model->b0;
      project_beliefs_recurse(solver.belief_tree->root);

      solver.calculate_initial_upper_bound();
      
      for(auto& bn : solver.belief_tree->nodes)
      {
        solver.insert_into_feature_tree(bn);
        bn->value_upper_bound = solver.simplex_upper_bound.dot(bn->b.p);
      }
    }

    void project_beliefs_recurse(belief_node_t* bn)
    {
      for(auto& ce : bn->children)
      {
        belief_t next_belief = model->next_belief(bn->b, ce->aid, ce->oid);
        ce->end->b = next_belief;

        project_beliefs_recurse(ce->end);
      }
    }
    
    void project_alpha_vectors()
    {
#if 0
      int ns = model->ns;
      for(auto& pa : solver.alpha_vectors)
      {
        int nsp = pa->grad.rows();
        vec t1(ns);
        t1.head(nsp) = pa->grad;
        for(int i=nsp; i<ns; i++)
          t1(i) = 0; 
        
        pa->grad = t1;
      }
#else
      for(auto& pa : solver.alpha_vectors)
        delete pa;
      solver.alpha_vectors.clear();
      solver.initialise_blind_policy();
#endif
    }
    
    void solver_iteration(int hw=100)
    {
      for(int i=0; i<hw; i++)
      {
        solver.iterate();

        if(hw >= 1){
          if(i%1 == 0){
            cout<<"("<< solver.belief_tree->root->value_upper_bound<<","<<
              solver.belief_tree->root->value_lower_bound<<") av: "<< solver.alpha_vectors.size() << " bn: "<< 
              solver.belief_tree->nodes.size()<<endl;
            //solver.print_alpha_vectors();
            //solver.print_belief_tree();
            //getchar();
          }
        }
        if(solver.is_converged())
          break;
      }
      cout<<"("<< solver.belief_tree->root->value_upper_bound<<","<<
      solver.belief_tree->root->value_lower_bound<<") av: "<< solver.alpha_vectors.size() << " bn: "<< 
      solver.belief_tree->nodes.size()<<endl;
    }

    void refine(int hws, int hwu, int hwo, int hw=1000)
    {
      create_model.refine(hws, hwu, hwo);
      project_alpha_vectors();
      
      solver.calculate_initial_upper_bound();
      project_beliefs();
      solver.prune_alpha();
      solver.update_bounds();

      solver_iteration(hw);
    }

    void solve(int hw=1000)
    {
      float convergence_threshold = 0.1;
      float insert_distance = 1e-6;
      solver.initialise(model->b0, model, insert_distance, convergence_threshold);
      solver_iteration(hw);
      
    }

};

#endif
