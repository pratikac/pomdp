#ifndef __system_h__
#define __system_h__

#include "../linalg.h"
#include "../utils.h"

#define RANDF   (rand()/(RAND_MAX+1.0))

template<size_t dim> class region_t;

template<size_t ns, size_t nu, size_t no>
class system_t{
  public:
    
    region_t<ns> operating_region, goal_region;
    region_t<nu> control_region;
    region_t<no> observation_region;
    vector<region_t<ns> > obstacles;
    
    system_t();
    ~system();

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
    
    virtual vec sample_state() = 0;
    virtual vec sample_control() = 0;
    virtual vec sample_observation() = 0;
    virtual vec sample_observation(const vec& s) = 0;
    virtual vec get_key(const vec& s) = 0;

    virtual vec get_fdt(const vec& s, const vec& u, float dt=1.0) = 0;
    virtual vec get_FFdt(const vec& s, const vec&u, float dt=1.0) = 0;
    virtual mat get_GG(const vec& s) = 0;
    virtual float get_ht(const vec& s, const vec& u, const float r) = 0;
};

template<size_t dim>
class region_t{
  public:
    vec c;
    vec s;
    region_t(vec& c_in, vec& s_in)
    {
      c = c_in;
      s = s_in;
    }
    bool is_inside(vec& state)
    {
      vec diff = (state - c)/2.0;
      for(int i=0; i<dim; i++)
      {
        if( (diff(i) > s(i)/2) || (diff(i) < -s(i)/2) )
          return false;
      }
      return true;
    }

};
#endif
