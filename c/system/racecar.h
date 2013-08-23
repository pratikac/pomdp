#ifndef __racecar_h__
#define __racecar_h__

#include "system.h"

class racecar_t : public system_t<6, 1, 3>
{
  public:
    typedef system_t<6,1,3> sys_t;
    using sys_t::operating_region;
    using sys_t::goal_region;
    using sys_t::control_region;
    using sys_t::observation_region;
    using sys_t::init_state;
    using sys_t::init_var;
    using sys_t::is_in_goal;
    using sys_t::is_in_obstacle;
    
    const static size_t ds = 6;
    const static size_t du = 1;
    const static size_t ddo = 3;
    vector<region_t<ds> > obstacles;
    
    float cfn, cfr, w, l, lf, lr, J, v, m;

    // states       : (x,y,beta,psi,r,cf)
    // controls     : (df)
    // observations : (x,y,psi)
    racecar_t()
    {
      cfn=1.5;
      cfr = 1.5;
      w=4;
      l=4;
      lf=l/2;
      lr = l/2;
      J = 1;
      m = 1;
      v = 1;

      vec opc(ds), ops(ds);
      opc << -3*w/2, -3*w/2, 0, 3*M_PI/4, 0, cfn;
      ops << 3*w, 3*w, M_PI/2, 3*M_PI/2, 5, 0.5;
      operating_region = region_t<ds> (opc, ops);

      vec ouc(du), ous(du);
      ouc << 0;
      ous << 1;
      control_region = region_t<du> (ouc, ous);

      vec ooc(ddo), oos(ddo);
      ooc << -3*w/2, -3*w/2, 3*M_PI/4;
      oos << 3*w, 3*w, 3*M_PI/2;
      observation_region = region_t<ddo> (ooc, oos);
     
      init_state = vec(ds);
      init_state << -w/4, -2.5*w, 0, M_PI/2, 0, cfr;
      
      float e1 = 0.01, e2 = 0.2;
      init_var = e1*mat::Identity(6,6);
      init_var(5,5) = e2;

      vec grc = opc, grs = ops;
      grc(0)= -2.5*w; grc(1) = -w/4;
      grs(0)= 1; grs(1) = 1;
      goal_region = region_t<ds> (grc, grs);
    }
    ~racecar_t(){};

    vec sample_state()
    {
      vec s(ds);
      region_t<ds>* r = &operating_region;
      float p = RANDF;
      if(p < 0.1)
      {
        r = &goal_region;
      }
      for(size_t i=0; i< ds; i++)
        s(i) = r->c(i) + r->s(i)*(RANDF-0.5);
      return s;
    }
    bool is_in_obstacle(vec& s)
    {
      return false;
    }
    bool is_in_obstacle(vec& s1, vec& s2)
    {
      return false;
    }
    vec sample_control()
    {
      vec u(du);
      region_t<du> r = control_region;
      for(size_t i=0; i< du; i++)
        u(i) = r.c(i) + r.s(i)*(RANDF-0.5);
      return u;
    }
    vec sample_observation()
    {
      vec o = sample_state();
      return get_observation(o);
    }
    vec sample_observation(vec& s)
    {
      return get_observation(s);
    }

    vec get_observation(const vec& s)
    {
      vec o(ddo);
      o << s(0), s(1), s(3);
      return o;
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
      vec f(ds);
      f <<  v*cos(s(3)+s(2)), 
            v*sin(s(2)+s(3)), 
            (-2*s(5)*s(2) - m*v*s(4))/m/v + s(5)/m/v*u(0),
            s(4),
            -1/J*s(5)*lf*l*s(4)/v + 1/J*s(5)*lf*u(0),
            0;
      return f*dt;
    }
    mat get_FFdt(const vec& s, const vec& u, float dt=1.0)
    {
      return init_var*dt;
    }
    mat get_GG(const vec& s)
    {
      return mat::Identity(ddo,ddo);
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
          return 0*-u.norm()*dt;
        else
          return -1000;
      }
      else
      {
        if(gs2)
          return 1000;
        else
          return 0*-u.norm()*dt;
      }
      return 0;
    }
};

#endif
