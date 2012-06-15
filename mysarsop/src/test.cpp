#include "pomdp.h"
#include "sarsop.h"

using namespace pomdp;
using namespace sarsop;

int main()
{
    //test_mymath();
    Model m = test_model();

    Solver s(m);
    
    return 0;
}
