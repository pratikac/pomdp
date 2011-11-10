#ifndef __common_h__
#define __common_h__

#include <iostream>
#include <cstdio>
#include <math.h>
#include <cfloat>
#include <list>
#include <vector>
#include <queue>
#include <fstream>
#include <cstdlib>
#include <sys/time.h>
#include <algorithm>
#include <utility>
#include <cassert>
#include <string>

#include <bot_core/bot_core.h>
#include <bot_lcmgl_client/lcmgl.h>
#include <bot_vis/gl_util.h>
#include "kdtree.h"

#include <Eigen/Dense>
using namespace Eigen;

using namespace std;

#define RANDF       (rand()/(RAND_MAX+1.0))


double randn();
void multivar_normal(double *mean, double *var, double *ret, int dim);
double sq(double x);
double normal_val(double *mean, double *var, double *tocalci, int dim);

inline
double sq(double x)
{
    return (x)*(x);
}

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
   
    if ( to_ret < 1e-30)
        to_ret = 1e-30;

    return to_ret;
}

void tic();
double toc();
double get_msec();

#endif

