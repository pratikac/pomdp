#ifndef __pomdp_h__
#define __pomdp_h__

#include <vector>
#include <cmath>
#include "mymath.h"
using namespace std;

namespace pomdp
{
    typedef vector< vector< vector<float> > > ttrans;
    typedef vector< vector< vector<float> > > tobs;
    typedef vector< vector<float> > treward;

    class Belief
    {
        public:
            vector<float> p;
            int dim;
            
            void normalize()
            {
                float sum = 0;
                for(int i=0; i<dim; i++)
                    sum += p[i];
                for(int i=0; i<dim; i++)
                    p[i] = p[i]/sum;
            }
    };
    class Model
    {
        public:
            int nstates;
            int nactions;
            int nobservations;
            float discount;
            
            /*! transition matrix
             * A x S x S : control x to x from
             */
            ttrans ptransition;
            /*! observation matrix
             * A x O x S : P(o | s, a)
             */
            tobs pobservation;
            /*! reward matrix
             * A x S : reward for taking control a at state s
             */
            treward preward;
            Belief b0;

            Model(int ns,int na, int no, ttrans& tin, tobs& oin, float din, treward& rin, Belief& bin)
            {
                nstates = ns;
                nactions = na;
                nobservations = no;
                discount = din;
                b0 = bin;

                ptransition = ttrans(na, vector< vector<float> >(ns, vector<float> (ns)));
                pobservation = tobs(na, vector< vector<float> >(no, vector<float> (ns)));
                preward = treward(na, vec(ns));

                for(int i=0; i<na; i++)
                {
                    for(int j=0; j<ns; j++)
                    {
                        for(int k=0; k<ns; k++)
                            ptransition[i][j][k] = tin[i][j][k];
                    }
                }
                for(int i=0; i<na; i++)
                {
                    for(int j=0; j<no; j++)
                    {
                        for(int k=0; k<ns; k++)
                            pobservation[i][j][k] = oin[i][j][k];
                    }
                }
                for(int i=0; i<na; i++)
                {
                    for(int j=0; j<ns; j++)
                        preward[i][j] = rin[i][j];
                }
            };
            void print()
            {
                cout<<"states: "<<nstates<<endl;
                cout<<"actions: "<<nactions<<endl;
                cout<<"transition_probabilities: "<<endl;
                for(int i=0; i<nactions; i++)
                {
                    cout<<"action_id: "<<i<<endl;
                    print_mat(ptransition[i]);
                }
                cout<<"observation_probabilities: "<<endl;
                for(int i=0; i<nactions; i++)
                {
                    cout<<"action_id: "<<i<<endl;
                    print_mat(pobservation[i]);
                }
                
                cout<<"initial_belief: ";
                print_vec(b0.p);
                cout<<"discount: "<<discount<<endl;
                cout<<"reward_function: "<<endl;
                for(int i=0; i<nactions; i++)
                {
                    cout<<"action_id: "<<i<<endl;
                    print_vec(preward[i]);
                }
            }

            Belief next_belief(Belief& b, int aid=-1, int oid=-1)
            {
                Belief newb;
                newb.dim = b.dim;
                newb.p = b.p;

                if(aid != -1)
                    newb.p = mat_vec(ptransition[aid], b.p);
                if( (oid != -1) && (aid != -1))
                {
                    newb.p = vec_vec_termwise(pobservation[aid][oid], newb.p);
                    newb.normalize();
                }
                return newb;
            }
            
            float get_expected_step_reward(Belief& b, int aid)
            {
                return dot(preward[aid], b.p);
            }
            float get_p_o_given_b(Belief& b, int aid, int oid)
            {
                return dot(pobservation[aid][oid], b.p);
            }
    };
};

pomdp::Model create_model();
void test_model(pomdp::Model& m);

#endif
