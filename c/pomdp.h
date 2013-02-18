#ifndef __pomdp_h__
#define __pomdp_h__

#include <iostream>
#include <vector>
#include <cmath>
#include "linalg.h"
using namespace std;

namespace pomdp{

  typedef vector< mat > ttrans;
  typedef vector< mat > tobs;
  typedef vector< vec > treward;

  class Belief{
    public:
      vec p;
      void normalize(){
        p = p/p.sum();
      }
      void print(){
        cout<<p<<endl;
      }
  };

  /*! Implements alpha vectors
   * that are gradients of the value function
   */
  class Alpha{
    public:
      int actionid;
      vec gradient;

      Alpha(){
      };
      /*! Constructor
       * @param[in] aid Action Id: index of optimal action associated with alpha vector
       * @param[in] gradin gradin(s) = alpha(s) for all states s in S_n
       */
      Alpha(int aid, vec& gradin)
      {
        actionid = aid;
        gradient = gradin;
      }
      /*! return value function as dot product of alpha with belief
       * @param[in] b belief at which value is calculated
       * \return float value dot(gradient, b)
       */
      float get_value(Belief& b)
      {
        return gradient.dot(b.p);
      }
  };

  /*! A POMDP model
   */
  class Model
  {
    public:
      int nstates;
      int nactions;
      int nobservations;
      float discount;

      /*! transition matrix
       * A x S_1 x S_2 : P(s_2 | s_1, a)
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

        ptransition = tin;
        pobservation = oin;
        preward = rin;
      };
      void print()
      {
        cout<<"states: "<<nstates<<endl;
        cout<<"actions: "<<nactions<<endl;
        cout<<"transition_probabilities: "<<endl;
        for(int i=0; i<nactions; i++)
        {
          cout<<"action_id: "<<i<<"-"<<ptransition[i]<<endl;
        }
        cout<<"observation_probabilities: "<<endl;
        for(int i=0; i<nactions; i++)
        {
          cout<<"action_id: "<<i<<"-"<<pobservation[i]<<endl;
        }

        cout<<"initial_belief: "<<b0.p<<endl;
        cout<<"discount: "<<discount<<endl;
        cout<<"reward_function: "<<endl;
        for(int i=0; i<nactions; i++)
          cout<<"action_id: "<<i<<"-"<<preward[i]<<endl;
      }

      Belief next_belief(Belief& b, int aid=-1, int oid=-1)
      {
        Belief newb;
        newb.p = b.p;
        
        //cout<<b.p<<endl;

        if(aid != -1)
          newb.p = ptransition[aid] * b.p;
        if( (oid != -1) && (aid != -1))
        {
          vec t1 = vec(newb.p);
          t1 = (t1.array() * mat(pobservation[aid]).col(oid).array()).matrix();
          newb.p = t1.sparseView();
          newb.normalize();
        }
        return newb;
      }
      /*! returns \sum_s R(s,a) b(s)
       * @param[in] Belief& b : belief at which action is taken
       * @param[in] int aid : action id of the action
       * @param[out] float reward
      */
      float get_expected_step_reward(Belief& b, int aid)
      {
        return preward[aid].dot(b.p);
      }
      float get_p_o_given_b(Belief& b, int aid, int oid)
      {
        return pobservation[aid].col(oid).dot(b.p);
      }
  };
};

pomdp::Model create_model();
void test_model(pomdp::Model& m);

#endif
