/*
 *  \file   Logger.cpp
 *
 *  \date   Oct 11, 2010
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#include "logger.h"

using namespace std;

int g_logDepth   = 2;
int g_depth      = 0;

SaturatedIndexedOutputStream lout(cout);
SaturatedIndexedOutputStream lerr(cerr);

using namespace std;

std::ostream& SaturatedIndexedOutputStream::operator<< (const os_manipulator& pf)
{
    if( g_depth > g_logDepth )
        return *this;

    if(pf == (os_manipulator)endl)
        m_flushed = true;
    m_stream << pf;
    return *this;
}
