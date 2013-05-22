#ifndef __pbvi_h__
#define __pbvi_h__

#include "utils.h"
#include "kdtree.h"

#include "pomdp.h"
#include "belief_tree.h"

#define RANDF   (rand()/(RAND_MAX+1.0))

class pbvi_t{
  public:
    typedef struct kdtree kdtree_t;
    typedef struct kdres kdres_t;

    model_t* model;
    belief_tree_t* belief_tree;
    kdtree_t* feature_tree;
    float insert_distance;

    set<alpha_t> alpha_vectors;

    pbvi_t(belief_t& b_root, model_t* model_in)
    {
      belief_tree = new belief_tree_t(b_root);
      model = model_in;
      feature_tree = kd_create(model->ns);
      insert_into_feature_tree(belief_tree->root);

      insert_distance = 0;
    }

    ~pbvi_t()
    {
      if(feature_tree)
        kd_free(feature_tree);
      if(belief_tree)
        delete belief_tree;
    }
    
    float* get_key(belief_node_t* bn)
    {
      return bn->b.p.data();
    }
    int insert_into_feature_tree(belief_node_t* bn)
    {
      float* key = get_key(bn);
      kd_insertf(feature_tree, key, bn);
      return 0;
    }

    bool check_insert_into_belief_tree(belief_node_t* par, edge_t* e)
    {
      belief_node_t* bn = e->end;
      kdres_t* kdres = kd_nearestf(feature_tree, get_key(bn));
      bool to_insert = false;
      if(kd_res_end(kdres))
        to_insert = true;
      else
      {
        belief_node_t* bnt = (belief_node_t*)kd_res_item_data(kdres);
        float tmp = bnt->b.distance(bn->b);
        //cout<<"tmp: "<< tmp << endl;
        if(tmp > insert_distance)
          to_insert = true;
      }
      kd_res_free(kdres);
      return to_insert;
    }
    int insert_into_belief_tree(belief_node_t* par, edge_t* e)
    {
      belief_tree->insert(par, e);
      insert_into_feature_tree(e->end);
      return 0;
    }

    edge_t* sample_child_belief(belief_node_t* par)
    {
      int aid = model->na*RANDF;
      int oid = model->no*RANDF;
      belief_t b = model->next_belief(par->b, aid, oid);
      belief_node_t* bn = new belief_node_t(b, par);
      return new edge_t(bn, aid, oid);
    }

    int sample_belief_nodes()
    {
      vector<pair<belief_node_t*, edge_t*> > nodes_to_insert;
      for(auto& bn : belief_tree->nodes)
      {
        edge_t* e_bn = sample_child_belief(bn);
        if(!check_insert_into_belief_tree(bn, e_bn))
          delete e_bn;
        else
          nodes_to_insert.push_back(make_pair(bn, e_bn));
      }
      for(auto& p : nodes_to_insert)
        insert_into_belief_tree(p.first, p.second);
      return nodes_to_insert.size();
    }

    int find_greater_alpha(alpha_t& a1, alpha_t& a2)
    {
      int tmp = 0;
      for(auto& bn : belief_tree->nodes)
      {
        float t1 = a1.get_value(bn->b);
        float t2 = a2.get_value(bn->b);
        if(t1 > t2)  
          tmp++;
        else if(t2 > t1)
          tmp--;
      }
      if(tmp == belief_tree->nodes.size())
        return 1;
      else if(tmp == -belief_tree->nodes.size())
        return -1;
      else
        return 0;
    }
    
    int backup(belief_node_t* bn)
    {
      int ns = model->ns;
      int no = model->no;
      int na = model->na;

      alpha_t* alpha_a_o[ns][no] = {nullptr};
      for(int a=0; a < na; a++)
      {
        for(int o=0; o< no; o++)
        {
          float max_val = -FLT_MAX;
          belief_t nb = model->next_belief(bn->b, a, o);
          for(auto& av : alpha_vectors)
          {
            float t1 = av.get_value(nb);
            if(t1 > max_val)
            {
              max_val = t1;
              alpha_a_o[a][o] = &av;
            }
          }
        }
      }
      
      alpha_t new_alpha[na];
      for(int a=0; a< na; a++)
      {
        new_alpha[a] = model->pr[a] + model->discount* 
      }
      
      return 0;
    }

    int backup_belief_nodes()
    {
      return 0;
    }

};


#endif
