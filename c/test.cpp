#define LOG_DEBUG

#include "sarsop.h"

using namespace pomdp;
using namespace sarsop;

int main()
{
    //test_mymath();
    
    Model m = create_model();
    Solver s = Solver(m);
    s.initialize();
    
    s.solve(0.1);
    
    return 0;
}
