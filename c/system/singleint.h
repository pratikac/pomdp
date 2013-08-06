#ifndef __singleint_h__
#define __singleint_h__

#include "system.h"

template<size_t ds, size_t du, size_t ddo>
class singleint_t : public system_t<ds, du, ddo>
{
  public:
    typedef system_t<ds,du,ddo> sys_t;
    using sys_t::operating_region;
    using sys_t::goal_region;
    using sys_t::control_region;
    using sys_t::observation_region;
    using sys_t::obstacles;
    using sys_t::init_state;
    using sys_t::init_var;
    using sys_t::is_in_goal;
    using sys_t::is_in_obstacle;
    
    singleint_t()
    {
      vec zero = vec::Zero(ds);
      vec two = mat::Constant(ds,1,2);
      operating_region = region_t<ds> (zero, two);
      control_region = region_t<du> (zero, two);
      observation_region = region_t<ddo> (zero, two);
      
      if(ds == 1)
      {
        init_state = vec(1); 
        init_state(0) = 0.9;
        init_var = mat(1,1); 
        init_var(0,0) = 0.1;
        goal_region = region_t<ds> (vec::Constant(ds,0), vec::Constant(ds,0.2));
      }
      else if(ds == 2)
      {
        init_state = vec(2); 
        init_state << 0.9, 0.9;
        init_var = mat(2,2);
        init_var << 0.1, 0, 0, 0.1;
        goal_region = region_t<ds> (vec::Constant(ds,0), vec::Constant(ds,0.2));
      }
      else
      {
        cout<<"dimension for singleint should be < 3"<<endl;
        abort();
      }
    }
    ~singleint_t(){};

    vec sample_state()
    {
      vec s = vec::Zero(ds);
      region_t<ds>* r = &operating_region;
      float p = RANDF;
      if(p < 0.2)
      {
        r = &goal_region;
      }
      for(size_t i=0; i< ds; i++)
        s(i) = r->c(i) + r->s(i)*(RANDF-0.5);
      return s;
    }
    vec sample_control()
    {
      vec u = vec::Zero(du);
      region_t<du> r = control_region;
      for(size_t i=0; i< ds; i++)
        u(i) = r.c(i) + r.s(i)*(RANDF-0.5);
      return u;
    }
    vec sample_observation()
    {
      vec o = sample_state();
      return get_observation(o);
    }
    vec get_observation(const vec& s)
    {
      return s;
    }
    vec get_key(const vec& s)
    {
      vec k = (s - operating_region.c);
      for(size_t i=0; i<ds; i++)
        k(i) = k(i)/operating_region.s(i) + 0.5;
      
      return k;
    }
    vec get_fdt(const vec& s, const vec& u, float dt=1.0)
    {
      return (-s + u)*dt;
    }
    vec get_FFdt(const vec& s, const vec& u, float dt=1.0)
    {
      return mat::Identity(ds,ds)*0.1;
    }
    mat get_GG(const vec& s)
    {
      return mat::Identity(ddo,ddo)*0.1;
    }
    float get_ht(const vec& s, const vec& u, const float r)
    {
      return r*r/(r*get_fdt(s,u,1).norm() + get_FFdt(s,u,1).norm());
    }
    
    float get_reward(vec& s1, vec& u, vec& s2, float dt)
    {
      bool gs1 = is_in_goal(s1);
      bool gs2 = is_in_goal(s2);

      if(gs1)
      {
        if(gs2)
          return 0.5*(-s1.norm() -u.norm())*dt;
        else
          return -1000;
      }
      else
      {
        if(gs2)
          return 1000;
        else
          return 0.5*(-s1.norm() -u.norm())*dt;
      }
      return 0;
    }
};

#endif