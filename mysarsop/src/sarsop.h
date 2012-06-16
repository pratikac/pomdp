#ifndef __sarsop_h__
#define __sarsop_h__

#include "pomdp.h"
#include "kdtree.h"

using namespace pomdp;

namespace sarsop{


    class Alpha
    {
        public:
            int actionid;
            vector<float> gradient;

            Alpha(){};
            Alpha(int aid, vector<float>& gradin)
            {
                actionid = aid;
                for(unsigned int i=0; i< gradin.size(); i++)
                    gradient.push_back(gradin[i]);
            }
            float get_value(Belief& b)
            {
                return dot(gradient, b.p);
            }
    };

    class BeliefNode
    {
        public:
            Belief b;

            BeliefNode* parent;
            vector<BeliefNode*> children;
            vector< pair<int, int> > aoid;
            int nchildren;

            float upper_bound, lower_bound;

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

                upper_bound = large_num;
                lower_bound = -large_num;
            }
            void get_key(double* k)
            {
                for(int i=0; i< b.dim; i++)
                    k[i] = b.p[i];
            }
    };
    class Solver
    {
        public:
            Model* model;
            vector<Alpha> alphas;
            vector<BeliefNode*> belief_tree_nodes;
            BeliefNode* root_node;

            struct kdtree* belief_tree;

            Solver(Model& min)
            {
                model = &min;
                belief_tree = kd_create(model->b0.dim);
                BeliefNode* b0 = new BeliefNode(model->b0, NULL, -1, -1);
                insert_belief_node_into_tree(b0);
                root_node = b0;
            }
            ~Solver()
            {
                kd_free(belief_tree);
                for(unsigned int i=0; i<belief_tree_nodes.size(); i++)
                    delete belief_tree_nodes[i];
                belief_tree_nodes.clear();
            }
            void insert_belief_node_into_tree(BeliefNode* b)
            {
                double* key = new double[model->b0.dim];
                b->get_key(key);
                kd_insert(belief_tree, key, b);
                delete key;
            }

            void backup(Belief* b);

            float predicted_optimal_reward(Belief& b);
            float lower_bound_reward(Belief& b);
            float upper_bound_reward(Belief& b);
            float get_bound_child(Belief& b, bool is_lower, int& aid);
            float get_poga_mult_bound(Belief& b, int aid, int oid, bool is_lower);
            void sample(float target_epsilon);
            void sample_belief_points(BeliefNode* bn, float L, float U, float epsilon, int level);
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
