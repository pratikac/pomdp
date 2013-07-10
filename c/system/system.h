#ifndef __system_h__
#define __system_h__

class system_t{
  public:
    /// dimension of states, controls, observations
    int ns, nu, no;

    system_t();
    ~system();
};

#endif
