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
    vector<vec> P(2, vec(2));
    vector<vec> Q(2, vec(2));
    vector<vec> R(3, vec(2));

    ttrans tmp1;
    tobs tmp2;

    // a1 = listen
    P[0][0] = 1; P[1][0] = 0.;
    P[0][1] = 0.; P[1][1] = 1;
    tmp1.push_back(P);

    // a2 = open left door
    P[0][0] = 0.5; P[1][0] = 0.5;
    P[0][1] = 0.5; P[1][1] = 0.5;
    tmp1.push_back(P);
    // a3 = open right door
    tmp1.push_back(P);
    
    // a1 = listen
    Q[0][0] = 0.85; Q[0][1] = 0.15;
    Q[1][0] = 0.15; Q[1][1] = 0.85;
    tmp2.push_back(Q);
    
    // a2 = open left door
    tmp2.push_back(P);
    // a2 = open right door
    tmp2.push_back(P);
    
    // reward function
    R[0][0] = -1; R[0][1] = -1;
    R[1][0] = -100; R[1][1] = 10;
    R[2][0] = 10; R[2][1] = -100;

    vec vb0(2); vb0[0]=0.5; vb0[1]=0.5;
    Belief b0;
    b0.dim = 2;
    b0.p = vb0; 
    Model m(2, 3, 2, tmp1, tmp2, 0.90, R, b0);
    
    //m.print();

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
