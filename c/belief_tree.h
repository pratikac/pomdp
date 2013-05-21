#ifndef __belief_tree_h__
#define __belief_tree_h__

#include "utils.h"
#include "linalg.h"
#include "pomdp.h"
#include "float.h"
#include <string.h>

class edge_t;
class belief_node_t;
class belief_tree_t;

class belief_node_t{
  public:
    belief_t b;
    float value_lower_bound, value_upper_bound;

    belief_node_t* parent;
    set<edge_t*> children;

    belief_node_t() : value_lower_bound(-FLT_MAX), value_upper_bound(FLT_MAX), parent(nullptr)
    {}
    
    belief_node_t(belief_t& b_in, belief_node_t* par=nullptr) : value_lower_bound(-FLT_MAX), value_upper_bound(FLT_MAX)
    {
      b = b_in;
      parent = par;
    }
    
    void print_tree()
    {
    }
};

class edge_t{
  public:
    belief_node_t* end;
    int aid;
    int oid;
    edge_t(belief_node_t* end_in, int aid_in, int oid_in)
      : end(end_in), aid(aid_in), oid(oid_in)
    {}
    ~edge_t()
    {
      delete end;
    }
};

class belief_tree_t{
  public:
    vector<belief_node_t*> nodes;
    belief_node_t* root;

    belief_tree_t() : root(nullptr)
    {}
    
    belief_tree_t(belief_t& b_root)
    {
      root = new belief_node_t(b_root);
      nodes.push_back(root);
    }

    ~belief_tree_t()
    {
      for(auto& bn : nodes)
        delete bn;
    }

    int insert(belief_node_t* par, edge_t* e)
    {
      nodes.push_back(e->end);
      e->end->parent = par;
      par->children.insert(e);
      return 0;
    }

    void print(belief_node_t* bn, string prefix="")
    {
      bn->b.print(prefix);
      prefix += "\t";
      for(auto& bne : bn->children)
        print(bne->end, prefix);
    }
};

#endif
