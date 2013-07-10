#ifndef __lightdark_h__
#define __lightdark_h__

#include "../system.h"

template<size_t ns, size_t nu, size_t no>
class lightdark_t : public system_t<ns, nu, no>
{
  public:
    vector<region_t<ns> > light_regions;

    lightdark_t()
    {
      ns = dim;
      nu = dim;
      no = dim;
    }
    
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
      for(int i=0; i< ns; i++)
        s(i) = r.c(i) + r.s(i)*(RANDF-0.5);
      return s;
    }
    vec sample_control()
    {
      vec u = Zero(nu);
      region_t<nu>& r = control_region;
      for(int i=0; i< ns; i++)
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
      for(int i=0; i<ns; i++)
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
