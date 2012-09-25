#include "mymath.h"

float dot(vec& v1, vec& v2)
{    
    int d = v1.size();
    float tmp = 0;
    for(int i=0; i<d; i++)
    {
        tmp = tmp + v1[i]*v2[i];
    }
    return tmp;
}

vec vec_vec_termwise(vec& v1, vec& v2)
{
    int d = v1.size();
    vector<float> tmp(d, 0);
    for(int i=0; i<d; i++)
    {
        tmp[i] = v1[i]*v2[i];
    }
    return tmp;
}

vec mat_vec(vector<vec>& mat, vec& vin)
{
    unsigned int d = vin.size();
    vector<float> tmp(d, 0);
    for(unsigned int i=0; i< mat.size(); i++)
    {
        if (mat[i].size() != d)
        {
            debug("mat_mult_error");
            exit(0);
        }
        tmp[i] = 0;
        for(unsigned int j=0; j< d; j++)
        {
            tmp[i] += mat[i][j]*vin[j];
        }
    }
    return tmp;
}

// mxn times nxk
vector< vec > mat_mat( vector<vec>& m1, vector<vec >& m2)
{
    if (m1[0].size() != m2.size())
    {
        debug("mat_mat_error");
        exit(0);
    }
    unsigned int n = m1[0].size();
    unsigned int m = m1.size();
    unsigned int k = m2[0].size();
    vector<vec > tmp(m, vec(k,0));

    for(unsigned int i=0; i< m; i++)
    {
        for(unsigned int j=0; j< k; j++)
        {
            tmp[i][j] = 0;
            for(unsigned int k=0; k<n; k++)
                tmp[i][j] += m1[i][k] * m2[k][j];
        }
    }
    return tmp;
}

void print_vec(vec& v)
{
    for(unsigned int i=0; i<v.size(); i++)
        cout<<v[i]<<" ";
    cout<<endl;
}
void print_mat(vector< vec >& m)
{
    for(unsigned int i=0; i< m.size(); i++)
    {
        cout<<"\t";
        for(unsigned int j=0; j< m[i].size(); j++)
            cout<<m[i][j]<<" ";
        cout<<endl;
    }
}
void print_mat(vector< vec_i >& m)
{
    for(unsigned int i=0; i< m.size(); i++)
    {
        cout<<"\t";
        for(unsigned int j=0; j< m[i].size(); j++)
            cout<<m[i][j]<<" ";
        cout<<endl;
    }
}

void test_mymath()
{
    vector<vec> P(2, vec(2));
    vector<vec> Q(2, vec(2));
    P[0][0] = 0.2; P[0][1] = 0.8;
    P[1][0] = 0.8; P[1][1] = 0.2;

    Q[0][0] = 0.6; Q[0][1] = 0.4;
    Q[1][0] = 0.4; Q[1][1] = 0.6;

    vec r(2);
    r[0] = 0.1; r[1]=0.2;

    vector<vec> tmp1 = mat_mat(P, Q);
    print_mat(tmp1);

    vec tmp2 = mat_vec(P, r);
    print_vec(tmp2);
}

inline
float sum(vec& v)
{
    float tmp = 0;
    for(unsigned int i=0; i< v.size(); i++)
        tmp += v[i];
    return tmp;
}
