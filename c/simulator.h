#ifndef __simulator_h__
#define __simulator_h__


#include "pomdp.h"


class simulator_t{
  public:
    vector<alpha_t*> alpha_vectors;
    model_t* m;

    simulator_t(model_t& model_in, vector<alpha_t>& alpha_vectors_in)
    {
      for(auto& av : alpha_vectors_in)
        alpha_vectors.push_back(&av);
      m = &model_in;
    }

    int find_best_action(belief_t& b)
    {
      float t1 = -FLT_MAX;
      int id = -1;
      for(auto& av : alpha_vectors)
      {
        float t2 = av->get_value(b);
        if(t2 > t1)
        {
          t1 = t2;
          id = av->aid;
        }
      }
      return id;
    }

    int simulate_one_step(int& sid, int& aid, int& oid, belief_t& b, float& rew, int& len)
    {
      aid = find_best_action(b);
      oid = m->sample_observation(b, aid);
      
      rew += pow(m->discount, len)*m->get_step_reward(sid, aid);
      len++;
      
      sid = m->next_state(sid, aid); 
      b = m->next_belief(b, aid, oid);
      return 0;
    }

    float simulate_trajectory(int steps)
    {
      belief_t b = m->b0;
      float rew=0;
      int len = 0, aid=-1, oid=-1;
      int sid = m->sample_state(b);
      for(int i=0; i< steps; i++)
      {
        simulate_one_step(sid, aid, oid, b, rew, len);
      }
      return rew;
    }
};

#endif
