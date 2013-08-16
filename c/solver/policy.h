#ifndef __policy_h__
#define __policy_h__


class policy_t{
  public:
    vector<alpha_t> alpha_vectors;

    policy_t(vector<alpha_t*>& av)
    {
      alpha_vectors.clear();
      for(auto& pav : av)
        alpha_vectors.push_back(*pav);
    }
};


#endif
