#include "mymath.h"

//################################
// Linear algebra
//################################

double sq(double x)
{
    return x*x;
}

double dot(vec& v1, vec& v2)
{    
    int d = v1.size();
    double tmp = 0;
    for(int i=0; i<d; i++)
    {
        tmp = tmp + v1[i]*v2[i];
    }
    return tmp;
}

vec vec_vec_termwise(vec& v1, vec& v2)
{
    int d = v1.size();
    vector<double> tmp(d, 0);
    for(int i=0; i<d; i++)
    {
        tmp[i] = v1[i]*v2[i];
    }
    return tmp;
}

vec mat_vec(vector<vec>& mat, vec& vin)
{
    unsigned int d = vin.size();
    vector<double> tmp(d, 0);
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

matrix mat_trans(matrix& m)
{
    matrix r = matrix(m[0].size(), zero(m.size()));
    for(unsigned int i=0; i<m.size(); i++)
    {
        for(unsigned int j=0; j< m[i].size(); j++)
        {
            r[j][i] = m[i][j];
        }
    }
    return r;
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

vector<vec > mat_double(matrix& m, double a)
{
    matrix r = matrix(m.size(), zero(m[0].size()));
    for(unsigned int i=0; i<m.size(); i++)
    {
        for(unsigned int j=0; j< m[i].size(); j++)
        {
            r[i][j] = m[i][j]*a;
        }
    }
    return r;
}

void print_vec(vec& v, ostream& stream, bool write_endl)
{
    for(unsigned int i=0; i<v.size(); i++)
        stream<<v[i]<<" ";
    if(write_endl)
        stream<<endl;
}
void print_mat(vector< vec >& m, ostream& stream)
{
    for(unsigned int i=0; i< m.size(); i++)
    {
        stream<<"\t";
        for(unsigned int j=0; j< m[i].size(); j++)
            stream<<m[i][j]<<" ";
        stream<<endl;
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
double sum(vec& v)
{
    double tmp = 0;
    for(unsigned int i=0; i< v.size(); i++)
        tmp += v[i];
    return tmp;
}

double norm_vec(vec& v)
{
    double t=0;
    for(unsigned int i=0; i< v.size(); i++)
        t = t+ v[i]*v[i];
    return sqrt(t);
}

double norm_mat(matrix& m)
{
    double sigma= -1;
    for(unsigned int i=0; i< m.size(); i++)
    {
        if(sigma < fabs(m[i][i]))
            sigma = fabs(m[i][i]);
    }
    return sigma;
}

state add(const state& s1, const state& s2)
{
    if(s1.size() != s2.size())
    {
        cout<<"add, size not same"<<endl;
        exit(0);
    }
    state s = zero(s1.size());
    for(unsigned int i=0; i< s1.size(); i++)
        s[i] = s1[i] + s2[i];
    return s;
}

state sub(const state& s1, const state& s2)
{
    if(s1.size() != s2.size())
    {
        cout<<"add, size not same"<<endl;
        exit(0);
    }
    state s = zero(s1.size());
    for(unsigned int i=0; i< s1.size(); i++)
        s[i] = s1[i] - s2[i];
    return s;
}

//###################################
// Probability
//##################################

// mean = 0, var = 1
double randn()
{
    static double x1, x2, w = 1.5;
    static bool which = 0;

    if(which == 1)
    {
        which = 0;
        return x2*w;
    }

    while (w >= 1.0){
        x1 = 2.0*RANDF - 1.0;
        x2 = 2.0*RANDF - 1.0;
        w = x1 * x1 + x2 * x2;
    }
    w = sqrt( (-2.0*log(w)) / w );
    
    which = 1;
    return x1 * w;
};

inline
double normal_val(double *mean, double *var, double *tocalci, int dim)
{
    double top = 0;
    double det = 1;
    for(int i=0; i<dim; i++)
    {
        top += sq(mean[i] - tocalci[i])/2/var[i];

        det = det*var[i];
    }
    top = exp(-0.5*top);
    double to_ret = (top/pow(2*M_PI, dim/2.0))/ sqrt( det );
    
    if ( to_ret < small_num)
        to_ret = small_num;

    return to_ret;
}

// for now, just write a wrapper
double normalval(vec& mu, matrix& sigma, vec& x)
{
    int ndim = mu.size();
    vec var = zero(ndim);
    for(int i=0; i<ndim; i++)
        var[i] = sigma[i][i];
    
    return normal_val(&(mu[0]), &(var[0]), &(x[0]), ndim);
}

void multivar_normal(double *mean, double *var, double *ret, int dim)
{
    for(int i=0; i < dim; i++)
        ret[i] = mean[i] + sqrt(var[i])*randn();
}



//###########################################
// Time
//###########################################
double curr_time;
void tic()
{
    struct timeval start;
    gettimeofday(&start, NULL);
    curr_time = start.tv_sec*1000 + start.tv_usec/1000.0;
}

double toc()
{
    struct timeval start;
    gettimeofday(&start, NULL);
    double delta_t = start.tv_sec*1000 + start.tv_usec/1000.0 - curr_time;
    
    cout<< delta_t/1000.0 << " [s]\n";
    return delta_t/1000.0;
}

double get_msec()
{
    struct timeval start;
    gettimeofday(&start, NULL);
    return start.tv_sec*1000 + start.tv_usec/1000.0;
}
