#define LOG_DEBUG

#include "sarsop.h"

using namespace pomdp;
using namespace sarsop;

int main()
{
    //test_mymath();
    Model m;
    Model m1 = m.test_model();

    Solver s(m1);
    
    return 0;
}
