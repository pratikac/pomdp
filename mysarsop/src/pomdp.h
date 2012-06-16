#ifndef __pomdp_h__
#define __pomdp_h__

#include <vector>
#include <cmath>
#include "mymath.h"
using namespace std;

namespace pomdp
{
    typedef vector< vector< vector<float> > > ttrans;
    typedef vector< vector<float> > tobs;
    typedef vector< vector<float> > treward;

    class Belief
    {
        public:
            vector<float> p;
            int dim;

            Belief()
            {
            };
            Belief(vec& pin)
            {
                p = pin;
                dim = pin.size();
            }
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

            ttrans ptransition;
            tobs pobservation;
            treward preward;
            Belief b0;

            Model(){};
            Model(int ns,int na, int no, ttrans& tin, tobs& oin, float din, treward& rin, const Belief& bin)
            {
                nstates = ns;
                nactions = na;
                nobservations = no;
                discount = din;
                b0 = bin;

                ptransition = ttrans(na, vector< vector<float> >(ns, vector<float> (ns)));
                pobservation = tobs(no, vector<float> (ns));
                preward = treward(ns, vector<float>(na));

                for(int i=0; i<na; i++)
                {
                    for(int j=0; j<ns; j++)
                    {
                        for(int k=0; k<ns; k++)
                            ptransition[i][j][k] = tin[i][j][k];
                    }
                }
                for(int i=0; i<no; i++)
                {
                    for(int j=0; j<ns; j++)
                    {
                        pobservation[i][j] = oin[i][j];
                    }
                }
                for(int i=0; i<ns; i++)
                {
                    for(int j=0; j<na; j++)
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
                cout<<"observations: "<<nobservations<<endl;
                cout<<"observation_probabilities: "<<endl;
                print_mat(pobservation);
                cout<<"initial_belief: ";
                print_vec(b0.p);
                cout<<"discount: "<<discount<<endl;
                cout<<"reward_function: "<<endl;
                print_mat(preward);
            }

            Belief next_belief(Belief& b, int aid=-1, int oid=-1)
            {
                Belief nb(b.p);
                if(aid != -1)
                    nb.p = mat_vec(ptransition[aid], nb.p);
                if(oid != -1)
                {
                    nb.p = vec_vec_termwise(pobservation[oid], nb.p);
                    nb.normalize();
                }
                return nb;
            }
            
            float get_expected_step_reward(Belief& b, int aid)
            {
                float tmp;
                for(int i=0; i< nstates; i++)
                    tmp += preward[i][aid]*b.p[i];
                return tmp;
            }
            float get_p_o_given_b(Belief& b, int oid)
            {
                float p_o_given_b = 0;
                for(int k=0; k< nstates; k++)
                {
                    p_o_given_b += pobservation[oid][k]*(b.p[k]);
                }
                return p_o_given_b;
            }
            /*! simple MDP model
             * s1, s2(goal), s3(terminal state)
             * a1 = transition, a2 = stopping action
             * observations noisy in first 2 states
             * reward in goal state
             */
            Model test_model()
            {
                vector<vec> P(3, vec(3));
                vector<vec> Q(3, vec(3));
                vector<vec> R(3, vec(2));
                
                ttrans tmp;
                
                // a1
                P[0][0] = 0.2; P[0][1] = 0.8; P[0][2] = 0;
                P[1][0] = 0.8; P[1][1] = 0.2; P[1][2] = 0;
                P[2][0] = 0.; P[2][1] = 0.; P[2][2] = 1;
                tmp.push_back(P);
                
                // a2
                P[0][0] = 0.; P[0][1] = 0.; P[0][2] = 1;
                P[1][0] = 0.; P[1][1] = 0.; P[1][2] = 1;
                P[2][0] = 0.; P[2][1] = 0.; P[2][2] = 1;
                tmp.push_back(P);

                Q[0][0] = 0.7; Q[0][1] = 0.3; Q[0][2] = 0;
                Q[1][0] = 0.3; Q[1][1] = 0.7; Q[1][2] = 0;
                Q[2][0] = 0.; Q[2][1] = 0.; Q[2][2] = 1;

                R[0][0] = -2; R[0][1] = -10;
                R[1][0] = -2; R[1][1] = 10;
                R[2][0] = 0; R[2][1] = 0;

                vec vb0(3); vb0[0]=0.9; vb0[1]=0.1; vb0[2] = 0.;
                Belief b0(vb0);
                Model m(3, 2, 3, tmp, Q, 0.75, R, b0);
                m.print();

                return m;
            }

    };

};

#endif
