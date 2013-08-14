#ifndef __belief_tree_h__
#define __belief_tree_h__

#include "utils.h"
#include "linalg.h"
#include "model.h"
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

    int depth,index;
    
    belief_node_t() : value_lower_bound(-FLT_MAX), value_upper_bound(FLT_MAX), 
          parent(nullptr), depth(0), index(-1)
    {}
    
    belief_node_t(belief_t& b_in, belief_node_t* par=nullptr) : value_lower_bound(-FLT_MAX), 
          value_upper_bound(FLT_MAX), depth(0), index(-1)
    {
      b = b_in;
      parent = par;
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
      root->index = 0;
      nodes.push_back(root);
    }

    ~belief_tree_t()
    {
      for(auto& bn : nodes)
        delete bn;
    }

    int insert(belief_node_t* par, edge_t* e)
    {
      e->end->index = nodes.size();
      nodes.push_back(e->end);
      e->end->parent = par;
      par->children.insert(e);
      return 0;
    }

    void print(belief_node_t* bn, string prefix="")
    {
      //bn->b.print(prefix, bn->value_lower_bound, bn->value_upper_bound);
      cout<<prefix<<bn<<" "<< bn->value_upper_bound<<", "<< bn->value_lower_bound<<endl;
      prefix += "\t";
      for(auto& bne : bn->children)
        print(bne->end, prefix);
    }

    bool all_children_are_leaves(belief_node_t* bn)
    {
      for(auto& ce : bn->children)
      {
        if(ce->end->children.size() > 0)
          return false;
      }
      return true;
    }

    int prune_below(belief_node_t* bn)
    {
      if(bn->parent)
      {
        for(auto ce = bn->parent->children.begin(); ce != bn->parent->children.end(); ce++)
        {
          if((*ce)->end == bn)
          {
            bn->parent->children.erase(ce);
            break;
          }
        }
      }
      prune(bn);
      return 0;
    }
    
    int prune(belief_node_t* bn)
    {
      for(auto& ce : bn->children)
        prune(ce->end);
      
      nodes.erase(nodes.begin()+bn->index);
      delete bn;
      
      return 0;
    }
};

#endif
