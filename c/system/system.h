#ifndef __system_h__
#define __system_h__

#include "../linalg.h"
#include "../utils.h"

#define RANDF   (rand()/(RAND_MAX+1.0))

template<size_t dim> class region_t;

template<size_t tds, size_t tdu, size_t tddo>
class system_t{
  public:
    
    const static int ds = tds;
    const static int du = tdu;
    const static int ddo = tddo;
    
    region_t<ds> operating_region, goal_region;
    region_t<du> control_region;
    region_t<ddo> observation_region;
    vector<region_t<ds> > obstacles;

    vec init_state;
    mat init_var;
  
    system_t(){};
    ~system_t(){};

    bool is_in_goal(vec& s)
    {
      return goal_region.is_inside(s);
    }

    bool is_in_obstacle(vec& s)
    {
      for(auto& r : obstacles)
      {
        if(r.is_inside(s))
          return true;
      }
      return false;
    }
    bool is_in_obstacle(vec& s1, vec& s2)
    {
      vec r;
      float t = 0;
      while(t < 1.1)
      {
        r = s1 + (s2-s1)*t;
        if(is_in_obstacle(r))
          return true;
        t += 0.1;
      }
      return false;
    }

    virtual vec sample_state() = 0;
    virtual vec sample_control() = 0;
    virtual vec zero_control() { return vec::Zero(du); }
    virtual vec sample_observation() = 0;
    virtual vec get_observation(const vec& s) = 0;
    virtual vec get_key(const vec& s) = 0;

    virtual vec get_fdt(const vec& s, const vec& u, float dt=1.0) = 0;
    virtual vec get_FFdt(const vec& s, const vec& u, float dt=1.0) = 0;
    virtual mat get_GG(const vec& s) = 0;
    virtual float get_ht(const vec& s, const vec& u, const float r) = 0;

    virtual float get_observation_prob(vec& s, vec& o)
    {
      // calculate P(o | s, a)
      mat GG = get_GG(s);
      vec os = get_observation(s);
      return normal_val(os, GG, o);
    }

    // R(s2 | s1, a)*dt
    virtual float get_reward(vec& s1, vec& u, vec& s2, float dt) = 0;
};

template<size_t dim>
class region_t{
  public:
    vec c;
    vec s;
    region_t(){};
    region_t(vec c_in, vec s_in)
    {
      c = c_in;
      s = s_in;
    }
    bool is_inside(const vec& state)
    {
      vec diff = (state - c)/2.0;
      for(size_t i=0; i<dim; i++)
      {
        if( (diff(i) > s(i)/2) || (diff(i) < -s(i)/2) )
          return false;
      }
      return true;
    }

};
#endif
