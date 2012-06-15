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
            vector<float> value;

            Alpha(int aid, vector<float>& valin)
            {
                actionid = aid;
                for(unsigned int i=0; i< valin.size(); i++)
                    value.push_back(valin[i]);
            }
    };

    class BeliefNode
    {
        public:
            Belief* b;
            BeliefNode* parent;
            vector<BeliefNode*> children;
            vector< pair<int, int> > aoid;
            int nchildren;

            BeliefNode(Belief& bin, BeliefNode* par)
            {
                b = &bin;
                if(par == NULL)
                    parent = NULL;
                else
                    parent = par;
                nchildren = 0;
            }
            void insert_child(BeliefNode& bn, int aid, int oid)
            {
                children.push_back(&bn);
                aoid.push_back( make_pair(aid, oid));
                nchildren++;
            }
            void get_key(double* k)
            {
                for(int i=0; i< b->dim; i++)
                    k[i] = b->p[i];
            }
    };
    class Solver
    {
        public:
            Model* m;
            vector<Alpha> alpha;
            vector<BeliefNode*> belief_tree_nodes;

            struct kdtree* belief_tree;

            Solver(Model& min)
            {
                m = &min;
                belief_tree = kd_create(m->b0.dim);
                BeliefNode* b0 = new BeliefNode(m->b0, NULL);
                insert_belief_node_into_tree(b0);
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
                double* key = new double[m->b0.dim];
                b->get_key(key);
                kd_insert(belief_tree, key, b);
                delete key;
            }
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
