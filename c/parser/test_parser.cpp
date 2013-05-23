#include "pomdp.h"
#include "mdp.h"

int main()
{
  model_t model;
  readMDP("../tiger.pomdp", model);
  writeMDP("tr.pomdp");

  model.print();

  return 0;
}
