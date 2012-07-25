#ifndef __sarsop_h__
#define __sarsop_h__

#include "utils.h"
#include "pomdp.h"
#include "kdtree.h"

using namespace pomdp;

namespace sarsop{

    /*! Implements alpha vectors
     * that are gradients of the value function
     */
    class Alpha
    {
        public:
            int actionid;
            vector<float> gradient;

            Alpha(){

            };
            /*! Constructor
             * @param[in] aid Action Id: index of optimal action associated with alpha vector
             * @param[in] gradin gradin(s) = alpha(s) for all states s in S_n
             */
            Alpha(int aid, vector<float>& gradin)
            {
                actionid = aid;
                for(unsigned int i=0; i< gradin.size(); i++)
                    gradient.push_back(gradin[i]);
            }
            /*! return value function as dot product of alpha with belief
             * @param[in] b belief at which value is calculated
             * \return float value dot(gradient, b)
             */
            float get_value(Belief& b)
            {
                return dot(gradient, b.p);
            }
     };

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

            float value_upper_bound, value_lower_bound, value_prediction_optimal;

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
            void get_key(double* k)
            {
                for(int i=0; i< b.dim; i++)
                    k[i] = b.p[i];
            }
            void print()
            {
                //cout<<"par: "; print_vec(parent->b.p);
                cout<<"belief prob: "; print_vec(b.p);
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
                belief_tree = kd_create(model->b0.dim);
                BeliefNode* b0 = new BeliefNode(model->b0, NULL, -1, -1);
                insert_belief_node_into_tree(b0);
                root_node = b0;

                mdp_value = vec(model->nstates, 0);
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
                double* key = new double[bn->b.dim];
                bn->get_key(key);
                kd_insert(belief_tree, key, bn);
                delete key;
            }
            void print_alphas()
            {
                for(unsigned int i=0; i<alphas.size(); i++)
                {
                    cout<<alphas[i].actionid<<": ";
                    print_vec(alphas[i].gradient);
                }
            }

            void mdp_value_iteration();
            void fixed_action_alpha_iteration();

            float get_predicted_optimal_reward(Belief& b);
            float get_lower_bound_reward(Belief& b);
            float get_upper_bound_reward(Belief& b);
            float get_bound_child(Belief& b, bool is_lower, int& aid);
            float get_poga_mult_bound(Belief& b, int aid, int oid, float& lower_bound, float& upper_bound);
            void sample(float target_epsilon);
            void sample_beliefs(BeliefNode* bn, float L, float U, float epsilon, int level);

            void backup(BeliefNode* bn);
            
            int check_alpha_dominated(Alpha& a1, Alpha& a2);
            int prune(bool only_last);

            void initialize();
            void solve(float target_epsilon);
            bool check_termination_condition(float ep);
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
