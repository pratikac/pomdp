#ifndef __mymath_h__
#define __mymath_h__

#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>
#include <sys/time.h>
#include "utils.h"
using namespace std;

#define large_num   1e10
#define small_num   1e-40

//################################
// Linear algebra
//################################

#define zero(x) vector<double>(x, 0)
typedef vector<double>           vec;
typedef vector< vector<double> > matrix;
typedef vector<int>             vec_i;
typedef vector< vector<int> >   matrix_i;
typedef vector<double>           state;

double dot(vec& v1, vec& v2);
vec vec_vec_termwise(vec& v1, vec& v2);

vec mat_vec(vector<vec>& mat, vec& vin);
vector< vec > mat_trans(vector<vec>& m);
vector< vec > mat_mat( vector<vec>& m1, vector<vec >& m2);
vector<vec > mat_double(matrix& m, double a);

void print_vec(vec& v, ostream& s=cout, bool write_endl=true);
void print_mat(vector< vec >& m, ostream& s=cout);
void print_mat(vector< vec_i >& m);
void test_mymath();

double sum(vec& v);
state add(const state& s1, const state& s2);
state sub(const state& s1, const state& s2);
double norm_vec(vec& v);
double norm_mat(matrix& m);

template <typename T>
int print_vec(vector<T>& s)
{
    for(unsigned int i=0; i<s.size(); i++)
        cout<<s[i]<<" ";
    cout<<endl;
    return 0;
}
template <typename T>
int print_matrix(vector< vector<T> >& m)
{
    for(unsigned int i=0; i<m.size(); i++)
    {
        for(unsigned int j=0; j< m[i].size(); j++)
            cout<<m[i][j]<<" ";
        cout<<endl;
    }
    return 0;
}


//###################################
// Probability
//##################################
#define RANDF       (rand()/(RAND_MAX+1.0))
double randn();
void multivar_normal(double *mean, double *var, double *ret, int dim);
double sq(double x);
double normal_val(double *mean, double *var, double *tocalci, int dim);
double normalval(vec& mu, matrix& sigma, vec& x);

//###########################################
// Time
//###########################################
void tic();
double toc();
double get_msec();
#endif
