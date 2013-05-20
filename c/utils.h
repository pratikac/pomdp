#ifndef __utils_h___
#define __utils_h___

#include <iostream>
#include <ctime>
#include <cstdio>
#include <vector>
#include <list>
#include <map>
#include <algorithm>
#include <set>
#include <queue>

#define debug(x) \
  std::cout<<"DBG("<<__FILE__<<":"<<__LINE__<<") "<<x<<std::endl
#define SQ(x)   (x)*(x)

typedef struct tt{
  std::clock_t _time;
  void tic()
  {
    _time = std::clock();
  }
  double toc() const
  {
    return double(std::clock() - _time)/CLOCKS_PER_SEC;
  }
}tt;

#endif
