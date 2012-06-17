#include "pomdp.h"

using namespace pomdp;

/*! simple MDP model
 * s1, s2(goal), s3(terminal state)
 * a1 = transition, a2 = stopping action
 * observations noisy in first 2 states
 * reward in goal state
 */
Model create_model()
{
    vector<vec> P(3, vec(3));
    vector<vec> Q(2, vec(2));
    vector<vec> R(2, vec(3));

    ttrans tmp1;
    tobs tmp2;

    // a1
    P[0][0] = 0.2; P[1][0] = 0.8; P[2][0] = 0;
    P[0][1] = 0.8; P[1][1] = 0.2; P[2][1] = 0;
    P[0][2] = 0.; P[1][2] = 0.; P[2][2] = 1;
    tmp1.push_back(P);

    // a2
    P[0][0] = 0.; P[1][0] = 0.; P[2][0] = 1;
    P[0][1] = 0.; P[1][1] = 0.; P[2][1] = 1;
    P[0][2] = 0.; P[1][2] = 0.; P[2][2] = 1;
    tmp1.push_back(P);
    
    Q[0][0] = 0.7; Q[0][1] = 0.3;
    Q[1][0] = 0.3; Q[1][1] = 0.7;
    tmp2.push_back(Q);
    tmp2.push_back(Q);

    R[0][0] = -2; R[1][0] = -10;
    R[0][1] = -2; R[1][1] = 10;
    R[0][2] = 0; R[1][2] = 0;

    vec vb0(3); vb0[0]=0.9; vb0[1]=0.1; vb0[2] = 0.;
    Belief b0;
    b0.dim = 3;
    b0.p = vb0; 
    Model m(3, 2, 2, tmp1, tmp2, 0.75, R, b0);
    m.print();

    return m;
}

void test_model(Model& m)
{
    // next_belief_testing
    Belief b1 = m.next_belief(m.b0, 0, 1);
    print_vec(b1.p);
    
    Belief b2 = m.next_belief(m.b0, 0, -1);
    print_vec(b2.p);
    b2 = m.next_belief(b2, -1, 1);
    print_vec(b2.p);
}
