#include "pomdp.h"

using namespace pomdp;

/*! simple MDP model
 * Tiger door problem (tony cassandra, technical report)
 * s1 = tiger_left, s2 = tiger_right
 * a1 = listen, a2 = open_left, a3 = open_right
 * o1 = left, o2 = right
 */
Model create_model()
{
  ttrans P = ttrans(3, spmat(2,2));
  tobs Q = tobs(3, spmat(2,2));
  treward R = treward(3, spvec(2));

  // a1 = listen
  P[0].insert(0,0) = 1; P[0].insert(1,0) = 0.;
  P[0].insert(0,1) = 0; P[0].insert(1,1) = 1.;

  // a2 = open left door
  P[1].insert(0,0) = 0.5; P[1].insert(1,0) = 0.5;
  P[1].insert(0,1) = 0.5; P[1].insert(1,1) = 0.5;
  // a3 = open right door
  P[2] = P[1];

  // a1 = listen
  Q[0].insert(0,0) = 0.85; Q[0].insert(0,1) = 0.15;
  Q[0].insert(1,0) = 0.15; Q[0].insert(1,1) = 0.85;

  // a2 = open left door
  Q[1] = P[1];
  // a2 = open right door
  Q[2] = P[1];

  // reward function
  R[0].insert(0) = -1; R[0].insert(1) = -1;
  R[1].insert(0) = -100; R[1].insert(1) = 10;
  R[2].insert(0) = 10; R[2].insert(1) = -100;

  spvec vb0(2); vb0.insert(0)=0.5; vb0.insert(1)=0.5;
  Belief b0;
  b0.p = vb0; 
  Model m(2, 3, 2, P, Q, 0.90, R, b0);

  //m.print();

  return m;
}

void test_model(Model& m)
{
  // next_belief_testing
  Belief b1 = m.next_belief(m.b0, 0, 1);
  cout<<b1.p<<endl;

  Belief b2 = m.next_belief(m.b0, 0, -1);
  cout<<b2.p<<endl;
  
  b2 = m.next_belief(b2, -1, 1);
  cout<<b2.p<<endl;
}
