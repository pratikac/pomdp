#ifndef __model_h__
#define __model_h__

#include <iostream>
#include <vector>
#include <cmath>
#include <string.h>
#include "linalg.h"
#include "utils.h"
using namespace std;

#define RANDF   (rand()/(RAND_MAX+1.0))

typedef vector<mat> pt_t;
typedef vector<mat> po_t;
typedef vector<mat> pr_t;

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
    void print(string prefix="", float val1=-FLT_MAX/2, float val2=-FLT_MAX/2) const
    {
      if(val1 > -FLT_MAX/2)
        cout<<prefix<<"["<<p.transpose()<<"]:"<<val1<<" "<< val2 << endl;
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
     * A x S x O : P(o | s, a)
     */
    po_t po;

    float discount;

    /*! reward matrix
     * A x S_1 x S_2 : R(s_2 | s_1, a) 
     */
    pr_t pr;
    belief_t b0;

    model_t(){
      ns = 0;
      na = 0;
      no = 0;

      pt.clear();
      po.clear();
      pr.clear();
      discount = -1;

      b0 = belief_t();
    }

    model_t(int ns_in,int na_in, int no_in, pt_t& pt_in, po_t& po_in, float d_in,
        pr_t& pr_in, belief_t& b0_in)
      : ns(ns_in), na(na_in), no(no_in), pt(pt_in), po(po_in), discount(d_in),
      pr(pr_in), b0(b0_in){
      }

    int normalize_mat()
    {
      for(size_t i=0; i<pt.size(); i++)
      {
        for(int j=0; j<ns; j++)
          pt[i].row(j) = pt[i].row(j)/pt[i].row(j).sum();
      }
      for(size_t i=0; i<po.size(); i++)
      {
        for(int j=0; j<ns; j++)
        {
          double t1 = po[i].row(j).sum();
          if(t1 < 1e-15)
          {
            cout<<"t1: "<< t1 << endl;
            assert(t1 > 1e-15);
          }
          po[i].row(j).noalias() = po[i].row(j)/t1;
        }
      }
      return 0;
    }

    void print(string fname="")
    {
      streambuf* buf;
      ofstream fout;

      if(fname == "")
      {
        buf = cout.rdbuf();
      }
      else
      {
        fout.open(fname);
        buf = fout.rdbuf();
      }
      ostream out(buf);

      out<<"states: "<<ns<<endl;
      out<<"actions: "<<na<<endl;
      out<<"observations: "<< no << endl;
      out<<"transition_probabilities: "<<endl;
      for(int i=0; i<na; i++)
      {
        out<<"aid: "<<i<<": ("<<pt[i].rows()<<","<<pt[i].cols()<<")"<<endl;
      }
      out<<"observation_probabilities: "<<endl;
      for(int i=0; i<na; i++)
      {
        out<<"aid: "<<i<<": ("<<po[i].rows()<<","<<po[i].cols()<<")"<<endl;
      }

      out<<"initial_belief: "<<b0.p.transpose()<<endl;
      out<<"discount: "<<discount<<endl;
      out<<"reward_function: "<<endl;
      for(int i=0; i<na; i++)
        out<<"aid: "<<i<<": ("<<pr[i].rows()<<","<<pr[i].cols()<<")"<<endl;

      if(fname != "")
        fout.close();
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

    belief_t delta_belief(int sid)
    {
      belief_t newb;
      newb.p = vec::Zero(ns);
      newb.p(sid) = 1;
      return newb;
    }

    mat get_p_a_o(int aid=-1, int oid=-1)
    {
      mat t1 = po[aid].col(oid).replicate(1,ns);
      return pt[aid].array() * (t1.transpose().array());
    }

    belief_t next_belief(belief_t& b, int aid=-1, int oid=-1)
    {
      belief_t newb(b);

      //cout<<b.p<<endl;

      if(aid != -1)
        newb.p = pt[aid] * b.p;
      if( (oid != -1) && (aid != -1))
      {
        mat t2 = po[aid];
        vec t1 = po[aid].col(oid);
        newb.p = newb.p.array() * t1.array();
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
      vec t1 = ((pr[aid].array())*(pt[aid].array())).rowwise().sum(); 
      return t1.dot(b.p);
    }

    float get_step_reward(const int& sid, const int& aid)
    {
      mat t1 = (pr[aid].array())*(pt[aid].array());
      return (t1.rowwise()).sum()(sid);
    }

    vec get_step_reward(const int& aid)
    {
      mat t1 = pr[aid].array()*pt[aid].array();
      vec t2 = t1.rowwise().sum().transpose();
      return t2; 
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

model_t create_example();
void test_model(model_t& m);

#endif
