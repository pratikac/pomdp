#include "pomdp.h"
#include "mdp.h"

int main()
{
  model_t model;
  readMDP("tiger.pomdp", model);
  return 0;
}
