/*
 *  \file   Logger.h
 *
 *  \date   Oct 11, 2010
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#ifndef __logger_h__
#define __logger_h__

#include <iostream>

extern int g_logDepth;
extern int g_depth;

static std::ostream& tab( std::ostream& stream )
{
    for( int i=0; i < g_depth; i++ )
        stream << "   ";
    return stream;
}

class SaturatedIndexedOutputStream : public std::ostream
{
    private:
        bool        m_flushed;
        std::ostream&    m_stream;

    public:
        SaturatedIndexedOutputStream(std::ostream& stream)
        :m_flushed(false), m_stream(stream){}

        template< typename T >
        SaturatedIndexedOutputStream& operator <<( const T& input )
        {
            if( g_depth > g_logDepth )
                return *this;

            if(m_flushed)
            {
                tab(m_stream);
                m_flushed = false;
            }

            m_stream << input;
            return *this;
        }

        typedef std::ostream& (*os_manipulator)(std::ostream&);

        std::ostream& operator<< (const os_manipulator& pf);


};

extern SaturatedIndexedOutputStream lout;
extern SaturatedIndexedOutputStream lerr;

class DepthTag
{
    public:
        DepthTag()
        {
            g_depth++;
        }

        ~DepthTag()
        {
            g_depth--;
        }
};



#endif
