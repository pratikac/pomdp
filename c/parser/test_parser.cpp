#include "model.h"
#include "mdp.h"

int main()
{
  model_t model;
  readMDP("../../examples/hallway.pomdp", model);
  writeMDP("tr.pomdp");

  model.print();

  return 0;
}
