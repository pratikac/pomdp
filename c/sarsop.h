#ifndef __sarsop_h__
#define __sarsop_h__

#include "utils.h"
#include "pomdp.h"
#include "kdtree.h"

using namespace pomdp;

namespace sarsop{

  const float large_num = 1e20;

  /*! node of a belief_tree, stores belief, pointers to parent, children, 
   * action-observation that result in the children edges, number of children
   * value function upper bound and lower bound
   */
  class BeliefNode
  {
    public:
      Belief b;

      BeliefNode* parent;
      vector<BeliefNode*> children;
      vector< pair<int, int> > aoid;
      int nchildren;

      double value_upper_bound, value_lower_bound, value_prediction_optimal;

      BeliefNode(Belief& bin, BeliefNode* par, int aid, int oid)
      {
        b = bin;
        if(par == NULL)
          parent = NULL;
        else
        {
          parent = par;
          par->children.push_back(this);
          par->nchildren++;
          par->aoid.push_back( make_pair(aid, oid));
        }
        nchildren = 0;

        value_upper_bound = large_num;
        value_lower_bound = -large_num;
        value_prediction_optimal = large_num;
      }
      /*! get key for belief of a node
       * note: the key is a pointer to the data in sparse vector
       * DONOT change the key before / after inserting in kdtree
       */
      void get_key(float* k)
      {
        k = vec(b.p).data();
      }
      void print()
      {
        //cout<<"par: "; print_vec(parent->b.p);
        cout<<"belief prob: "<< b.p<<endl;
        //cout<<"bounds: "<<value_upper_bound<<" "<<value_prediction_optimal<<" "<<value_lower_bound<<endl;
      }
  };
  class Solver
  {
    public:
      Model* model;

      vector<BeliefNode*> belief_tree_nodes;
      BeliefNode* root_node;
      struct kdtree* belief_tree;

      vec mdp_value;
      vector<Alpha> alphas;


      Solver(Model& min)
      {
        model = &min;
        belief_tree = kd_create(mode->nstates);
        BeliefNode* b0 = new BeliefNode(model->b0, NULL, -1, -1);
        insert_belief_node_into_tree(b0);
        root_node = b0;

        mdp_value = vec::Zero(model->nstates);
      }
      ~Solver()
      {
        kd_free(belief_tree);
        for(unsigned int i=0; i<belief_tree_nodes.size(); i++)
          delete belief_tree_nodes[i];
        belief_tree_nodes.clear();
      }
      void insert_belief_node_into_tree(BeliefNode* bn)
      {
        float* key;
        bn->get_key(key);
        kd_insertf(belief_tree, key, bn);
        delete key;
      }
      void print_alphas()
      {
        for(unsigned int i=0; i<alphas.size(); i++)
        {
          cout<<alphas[i].actionid<<": "<<alphas[i].gradient<<endl;
        }
      }

      void mdp_value_iteration();
      /*
      void fixed_action_alpha_iteration();

      double get_predicted_optimal_reward(Belief& b);
      double get_lower_bound_reward(Belief& b);
      double get_upper_bound_reward(Belief& b);
      double get_bound_child(Belief& b, bool is_lower, int& aid);
      double get_poga_mult_bound(Belief& b, int aid, int oid, double& lower_bound, double& upper_bound);
      void sample(double target_epsilon);
      void sample_beliefs(BeliefNode* bn, double L, double U, double epsilon, int level);

      void backup(BeliefNode* bn);

      int check_alpha_dominated(Alpha& a1, Alpha& a2);
      int prune(bool only_last);

      void initialize();
      void solve(double target_epsilon);
      bool check_termination_condition(double ep);
      */
  };

  class Simulator
  {
    public:
      Solver* solver;

      Simulator(Solver& s)
      {
        solver = &s;
      }
  };
};

#endif
