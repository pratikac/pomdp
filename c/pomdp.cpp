#include "pomdp.h"

/*! simple MDP model
 * Tiger door problem (tony cassandra, technical report)
 * s1 = tiger_left, s2 = tiger_right
 * a1 = listen, a2 = open_left, a3 = open_right
 * o1 = left, o2 = right
 */
model_t create_example()
{
  pt_t P = pt_t(3, mat(2,2));
  po_t Q = po_t(3, mat(2,2));
  pr_t R = pr_t(3, mat(2,2));

  // a1 = listen
  P[0](0,0) = 1; P[0](1,0) = 0.;
  P[0](0,1) = 0; P[0](1,1) = 1.;

  // a2 = open left door
  P[1](0,0) = 0.5; P[1](1,0) = 0.5;
  P[1](0,1) = 0.5; P[1](1,1) = 0.5;
  // a3 = open right door
  P[2] = P[1];

  // a1 = listen
  Q[0](0,0) = 0.85; Q[0](0,1) = 0.15;
  Q[0](1,0) = 0.15; Q[0](1,1) = 0.85;

  // a2 = open left door
  Q[1] = P[1];
  // a2 = open right door
  Q[2] = P[1];

  // reward function
  R[0] = -1*mat::Ones(2,2);
  R[1](0,0) = -100; R[1](0,1) = -100;
  R[1](1,0) = 10; R[1](1,1) = 10;
  R[2](0,0) = 10; R[2](0,1) = 10;
  R[2](1,0) = -100; R[2](1,1) = -100;

  vec vb0(2); vb0(0)=0.5; vb0(1)=0.5;
  belief_t b0;
  b0.p = vb0; 
  model_t m(2, 3, 2, P, Q, 0.95, R, b0);

  //m.print();

  return m;
}

void test_model(model_t& m)
{
  // next_belief_testing
  belief_t b1 = m.next_belief(m.b0, 0, 1);
  b1.print("b1");

  belief_t b2 = m.next_belief(m.b0, 0, -1);
  b2.print("b2");

  b2 = m.next_belief(b2, -1, 1);
  b2.print("b2");
}
