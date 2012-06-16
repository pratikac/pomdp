#ifndef __mymath_h__
#define __mymath_h__

#include <cstdlib>
#include <iostream>
#include <vector>
#include "utils.h"
using namespace std;

#define large_num   1e20

typedef vector<float> vec;
typedef vector<int> vec_i;
typedef vector< vector<int> > matrix_i;
typedef vector< vector<float> > matrix_f;


float dot(vec& v1, vec& v2);
vec vec_vec_termwise(vec& v1, vec& v2);
vec mat_vec(vector<vec>& mat, vec& vin);
vector< vec > mat_mat( vector<vec>& m1, vector<vec >& m2);
void print_vec(vec& v);
void print_mat(vector< vec >& m);
void test_mymath();

#endif
