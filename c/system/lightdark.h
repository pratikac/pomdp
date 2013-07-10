#ifndef __lightdark_h__
#define __lightdark_h__

#include "system.h"

template<size_t ns, size_t nu, size_t no>
class lightdark_t : public system_t<ns, nu, no>
{
  public:
    typedef system_t<ns,nu,no> sys_t;
    using sys_t::operating_region;
    using sys_t::goal_region;
    using sys_t::control_region;
    using sys_t::observation_region;
    using sys_t::obstacles;
    using sys_t::init_state;
    using sys_t::init_var;

    vector<region_t<ns> > light_regions;

    lightdark_t()
    {
      vec zero = Zero(ns);
      vec two = mat::Constant(ns,1,2);
      sys_t::operating_region = region_t<ns> (zero, two);
      control_region = region_t<nu> (zero, two);
      observation_region = region_t<no> (zero, two);

      if(ns == 1)
      {
        init_state = vec(1); 
        init_state(0) = 0;
        init_var = mat(1,1); 
        init_var(0,0) = 0.1;
        goal_region = region_t<ns> ((vec)mat::Constant(ns,1,-0.9), (vec)mat::Constant(ns,1,0.2));
        light_regions.push_back(region_t<ns> ((vec)mat::Constant(ns,1,0.9), (vec)mat::Constant(ns,1,0.2)));
      }
      else if(ns == 2)
      {

      }
      else
      {
        cout<<"dimension for lightdark should be < 3"<<endl;
        abort();
      }
    }
    ~lightdark_t(){};

    vec sample_state()
    {
      vec s = Zero(ns);
      region_t<ns>& r = operating_region;
      float p = RANDF;
      if(p < 0.1)
      {
        r = goal_region;
      }
      else if(p < 0.4)
      {
        int p2 = RANDF*light_regions.size();
        r = light_regions[p2];
      }
      for(size_t i=0; i< ns; i++)
        s(i) = r.c(i) + r.s(i)*(RANDF-0.5);
      return s;
    }
    vec sample_control()
    {
      vec u = Zero(nu);
      region_t<nu>& r = control_region;
      for(size_t i=0; i< ns; i++)
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
      vec k = s - operating_region.c;
      for(size_t i=0; i<ns; i++)
        k(i) = k(i)/operating_region.s(i) + 0.5;
      return k;
    }
    vec get_fdt(const vec& s, const vec& u, float dt=1.0)
    {
      return u*dt;
    }
    vec get_FFdt(const vec& s, const vec& u, float dt=1.0)
    {
      return Identity(ns,ns)*0.1;
    }
    mat get_GG(const vec& s)
    {
      bool is_light = false;
      for(auto& r : light_regions)
      {
        if(r.is_inside(s))
        {
          is_light = true;
          break;
        }
      }
      if(is_light)
        return Identity(no,no)*0.01;
      else
        return Identity(no,no)*100;
    }
    float get_ht(const vec& s, const vec& u, const float r)
    {
      return r*r/(r*get_fdt(s,u,1).norm() + get_FFdt(s,u,1).norm());
    }

};

#endif
