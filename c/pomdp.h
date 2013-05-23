#ifndef __pomdp_h__
#define __pomdp_h__

#include <iostream>
#include <vector>
#include <cmath>
#include "linalg.h"
#include <string.h>
using namespace std;

#define RANDF   (rand()/(RAND_MAX+1.0))

typedef vector<mat> pt_t;
typedef vector<mat> po_t;
typedef vector<vec> pr_t;

/*! beliefs
*/
class belief_t{
  public:
    vec p;
  
    belief_t(){}
    belief_t(const belief_t& b_in) : p(b_in.p){}

    void normalize()
    {
      p = p/p.sum();
    }
    void print(string prefix="", float val=-FLT_MAX/2) const
    {
      if(val > -FLT_MAX/2)
        cout<<prefix<<"["<<p.transpose()<<"]:"<<val<<endl;
      else 
        cout<<prefix<<"["<<p.transpose()<<"]"<<endl;
    }

    float distance(const belief_t& b) const
    {
      vec tmp = p-b.p;
      return tmp.array().abs().sum();
    }

    float entropy(){
      return -p.dot(p.array().log().matrix());
    }
};
/*! implements alpha vectors
 * that are gradients of the value function
 */
class alpha_t{
  public:
    int aid;
    vec grad;
    bool mark;

    alpha_t(): mark(0){
    };
    /*! Constructor
     * @param[in] aid Action Id: index of optimal action associated with alpha vector
     * @param[in] gradin gradin(s) = alpha(s) for all states s in S_n
     */
    alpha_t(int aid_in, vec& grad_in) : mark(0)
    {
      aid = aid_in;
      grad = grad_in;
    }

    const alpha_t operator-(const alpha_t& a2) const
    {
      alpha_t toret = *this;
      toret.grad -= a2.grad;
      toret.aid = -1;
      return toret;
    }
    /*! return value function as dot product of alpha with belief
     * @param[in] b belief at which value is calculated
     * \return float value dot(gradient, b)
     */
    float get_value(belief_t& b) const
    {
      return grad.dot(b.p);
    }

    void print(string prefix="") const
    {
      cout<<"aid: "<<aid<<"-["<<grad.transpose()<<"]"<<endl;
    }
};

/*! A POMDP model
*/
class model_t
{
  public:
    int ns;
    int na;
    int no;

    /*! transition matrix
     * A x S_1 x S_2 : P(s_2 | s_1, a)
     */
    pt_t pt;
    /*! observation matrix
     * A x O x S : P(o | s, a)
     */
    po_t po;

    float discount;

    /*! reward matrix
     * A x S : reward for taking control a at state s
     */
    pr_t pr;
    belief_t b0;
    
    model_t(){
    }
    
    model_t(int ns_in,int na_in, int no_in, pt_t& pt_in, po_t& po_in, float d_in,
        pr_t& pr_in, belief_t& b0_in)
      : ns(ns_in), na(na_in), no(no_in), pt(pt_in), po(po_in), discount(d_in),
      pr(pr_in), b0(b0_in){
    }
    void print()
    {
      cout<<"states: "<<ns<<endl;
      cout<<"actions: "<<na<<endl;
      cout<<"transition_probabilities: "<<endl;
      for(int i=0; i<na; i++)
      {
        cout<<"action_id: "<<i<<"-"<<pt[i]<<endl;
      }
      cout<<"observation_probabilities: "<<endl;
      for(int i=0; i<na; i++)
      {
        cout<<"action_id: "<<i<<"-"<<po[i]<<endl;
      }

      cout<<"initial_belief: "<<b0.p<<endl;
      cout<<"discount: "<<discount<<endl;
      cout<<"reward_function: "<<endl;
      for(int i=0; i<na; i++)
        cout<<"action_id: "<<i<<"-"<<pr[i]<<endl;
    }
    
    int multinomial_sampling(const vec& arr)
    {
      float r = RANDF;
      int len = arr.size();
      vector<float> t2;
      t2.assign(arr.data(), arr.data()+len);
      if(t2[0] > r)
        return 0;
      for(int i=1; i< len; i++)
      {
        t2[i] += t2[i-1];
        if(t2[i] > r)
          return i;
      }
      return len-1;
    }
    
    int next_state(const int& sid, const int& aid)
    {
      vec t1 = pt[aid].row(sid).transpose();
      return multinomial_sampling(t1);
    }

    belief_t next_belief(belief_t& b, int aid=-1, int oid=-1)
    {
      belief_t newb(b);

      //cout<<b.p<<endl;

      if(aid != -1)
        newb.p = pt[aid] * b.p;
      if( (oid != -1) && (aid != -1))
      {
        newb.p = newb.p.array() * po[aid].col(oid).array();
        newb.normalize();
      }
      return newb;
    }
    /*! returns \sum_s R(s,a) b(s)
     * @param[in] belief_t& b : belief at which action is taken
     * @param[in] int aid : action id of the action
     * @param[out] float reward
     */
    float get_expected_step_reward(const belief_t& b, const int& aid)
    {
      return pr[aid].dot(b.p);
    }
    
    float get_step_reward(const int& sid, const int& aid)
    {
      return pr[aid](sid);
    }

    float get_p_o_given_b(const belief_t& b, const int& aid, const int& oid)
    {
      return po[aid].col(oid).dot(b.p);
    }
    
    int sample_state(const belief_t& b)
    {
      return multinomial_sampling(b.p);
    }

    int sample_observation(const belief_t& b, const int& aid)
    {
      int sid = sample_state(b); 
      vec t1 = po[aid].row(sid).transpose();
      return multinomial_sampling(t1);
    }
    
    int sample_observation(const int& sid, const int& aid)
    {
      vec t1 = po[aid].row(sid).transpose();
      return multinomial_sampling(t1);
    }
};

/*! simple MDP model
 * Tiger door problem (tony cassandra, technical report)
 * s1 = tiger_left, s2 = tiger_right
 * a1 = listen, a2 = open_left, a3 = open_right
 * o1 = left, o2 = right
 */
model_t create_example()
{
  pt_t P = pt_t(3, mat(2,2));
  po_t Q = po_t(3, mat(2,2));
  pr_t R = pr_t(3, vec(2));

  // a1 = listen
  P[0](0,0) = 1; P[0](1,0) = 0.;
  P[0](0,1) = 0; P[0](1,1) = 1.;

  // a2 = open left door
  P[1](0,0) = 0.5; P[1](1,0) = 0.5;
  P[1](0,1) = 0.5; P[1](1,1) = 0.5;
  // a3 = open right door
  P[2] = P[1];

  // a1 = listen
  Q[0](0,0) = 0.85; Q[0](0,1) = 0.15;
  Q[0](1,0) = 0.15; Q[0](1,1) = 0.85;

  // a2 = open left door
  Q[1] = P[1];
  // a2 = open right door
  Q[2] = P[1];

  // reward function
  R[0](0) = -1; R[0](1) = -1;
  R[1](0) = -100; R[1](1) = 10;
  R[2](0) = 10; R[2](1) = -100;

  vec vb0(2); vb0(0)=0.5; vb0(1)=0.5;
  belief_t b0;
  b0.p = vb0; 
  model_t m(2, 3, 2, P, Q, 0.95, R, b0);

  //m.print();

  return m;
}

void test_model(model_t& m)
{
  // next_belief_testing
  belief_t b1 = m.next_belief(m.b0, 0, 1);
  b1.print("b1");

  belief_t b2 = m.next_belief(m.b0, 0, -1);
  b2.print("b2");

  b2 = m.next_belief(b2, -1, 1);
  b2.print("b2");
}

#endif
