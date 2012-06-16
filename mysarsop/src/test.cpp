#define LOG_DEBUG

#include "sarsop.h"

using namespace pomdp;
using namespace sarsop;

int main()
{
    //test_mymath();
    Model m;
    Model m1 = m.test_model();
    
    Solver s = Solver(m1);
    //s.mdp_value_iteration();
    s.fixed_action_alpha_iteration();

    return 0;
}
