#include "common.h"

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

void multivar_normal(double *mean, double *var, double *ret, int dim)
{
    for(int i=0; i < dim; i++)
        ret[i] = mean[i] + sqrt(var[i])*randn();
}

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

