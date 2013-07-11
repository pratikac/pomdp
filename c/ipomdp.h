#ifndef __ipomdp_h__
#define __ipomdp_h__

#include "model.h"
#include "pbvi.h"
#include "system/lightdark.h"

template<class system_t, class solver_t>
class ipomdp_t{
  public:
    int ns, nu, no;

    system_t system;
    solver_t solver;

    ipomdp_t(){
      ns = 10;
      nu = 2.0*log(ns);
      no = nu;
    }
    ~ipomdp_t(){};

};

#endif
