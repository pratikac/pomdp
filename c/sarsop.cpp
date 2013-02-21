#include "sarsop.h"

using namespace sarsop;

/*! basic value iteration for MDP
 * initializes upper bound of value function
 */
void Solver::mdp_value_iteration()
{
  Model& m = *model;

  vec mdp_value_copy = mdp_value;
  float epsilon = 1e-2;
  bool is_converged = false;
  while(!is_converged)
  {
    mdp_value_copy = mdp_value;

    is_converged = true;
    for(int i=0; i< m.nstates; i++)
    {
      float max_value = -large_num;
      for(int j=0; j< m.nactions; j++)
      {
        float tmp = m.discount * (mat(m.ptransition[j]).col(i).dot(mdp_value_copy));
        tmp = tmp + m.preward[j](i);
        if (tmp > max_value)
          max_value = tmp;
      }
      if( fabs(max_value - mdp_value_copy[i]) > epsilon)
        is_converged = false;
      mdp_value(i) = max_value;
    }
  }
  cout<<"mdp_value: "<<mdp_value.transpose()<<endl;
}

/*! calculates one single alpha plane for the best fixed action policy (HSVI2 paper)
*/
void Solver::fixed_action_alpha_iteration()
{
  Model& m = *model;

  Alpha alpha;
  alpha.gradient = vec(m.nstates);

  int fixed_aid = -1;
  float min_reward = -large_num;
  for(int j=0; j< m.nactions; j++)
  {
    float t1 = m.preward[j].minCoeff();
    if(t1 > min_reward)
    {
      min_reward = t1;
      fixed_aid = j;
    }
  }
  alpha.actionid = fixed_aid;
  for(int i=0; i< m.nstates; i++)
    alpha.gradient(i) = min_reward/(1.0 - m.discount);
  alphas.push_back(alpha);
  print_alphas();
}

/*! performs Bellman backup upstream (until root) of the argument belief node
 * @param[in] BeliefNode* bn : belief node to perform backup on
 */
void Solver::backup(BeliefNode* bn)
{
  Model& m = *model;
  int na = m.nactions;
  int no = m.nobservations;
  int ns = m.nstates;
  Belief& b = bn->b;
  mat alpha_ao_id = mat(na, no);

  for(int i=0; i<na; i++)
  {
    for(int j=0; j<no; j++)
    {
      int max_alpha_id = -1;
      float max_val = -large_num;
      for(unsigned int k=0; k < alphas.size(); k++)
      {
        Belief tmp_belief = m.next_belief(b, i, j);
        float tmp = alphas[k].get_value(tmp_belief);
        if(tmp > max_val)
        {
          max_val = tmp;
          max_alpha_id = k;
        }
      }
      alpha_ao_id(i,j) = (float)max_alpha_id;
    }
  }

  Alpha new_alpha, tmp_alpha;
  float max_val = -large_num;
  for(int i=0; i<na; i++)
  {
    mat alpha_ao(ns, no);
    for(int j=0; j<no; j++)
      alpha_ao.col(j) = alphas[alpha_ao_id(i,j)].gradient;
    // FIXME
    vec tmp = (m.pobservation[i].transpose() * alpha_ao).diagonal();
    tmp_alpha.gradient = m.preward[i] + m.ptransition[i]*tmp;
    tmp_alpha.actionid = i;

    float tmp_alpha_val = tmp_alpha.get_value(b);
    if( tmp_alpha_val > max_val)
      new_alpha = tmp_alpha;
  }

  alphas.push_back(new_alpha);
  cout<<"Inserted: "<<new_alpha.actionid<<" "<<new_alpha.gradient.transpose()<<" alpha.size: "<< alphas.size()<<endl;
}

/*! predict optimal reward using binning of beliefs
*/
float Solver::get_predicted_optimal_reward(BeliefNode* bn)
{
  double* key = new double[2];
  key[0] = bn->b.entropy();
  key[1] = get_mdp_upper_bound_reward(bn->b);
  struct kdres* res = kd_nearest(belief_tree, key);
  float mean_val = 0;
  float toret = 0;
  if(kd_res_size(res) < 2)
  {
    toret = key[1];
  }
  else
  {
    while(!kd_res_end(res))
    {
      BeliefNode* bn2 = (BeliefNode*) kd_res_item_data(res);
      if(bn != bn2)
      {
        mean_val += bn2->value_lower_bound;
      }
      kd_res_next(res);
    }
    toret = mean_val/(float)kd_res_size(res);
  }
  delete[] key;
  kd_res_free(res);
  return toret;
}

/*! lower bound using alpha vectors
 * @param[in] : Belief& b
 * @param[out] ; value at belief b calculating using alpha vectors
*/
float Solver::get_lower_bound_reward(Belief& b)
{
  float max_value = -large_num;
  for(unsigned int i=0; i< alphas.size(); i++)
  {
    float tmp = alphas[i].get_value(b);
    if(tmp > max_value)
      max_value = tmp;
  }
  return max_value;
}

/*! get upper / lower bound by expanding for one step
 * @param[in] : Belief& b
 * @param[in] : bool is_lower : upper bound / lower bound
 * @param[in] : action id (used only if calculating using a specific action
 * @param[out] : maximum value
 */
float Solver::get_bound_child(Belief& b, bool is_lower_bound, int& aid)
{
  Model& m = *model;

  float max_val = -large_num;
  for(int i=0; i< m.nactions; i++)
  {
    float poga_mult_bound = 0;
    for(int j=0; j< m.nobservations; j++)
    {
      float v_next_belief = 0;
      Belief new_belief_tmp = m.next_belief(b, i, j);
      if(is_lower_bound == true)
        v_next_belief = get_lower_bound_reward(new_belief_tmp);
      else
        v_next_belief = get_mdp_upper_bound_reward(new_belief_tmp);

      float p_o_given_b = m.get_p_o_given_b(b, i, j);
      poga_mult_bound = poga_mult_bound + (p_o_given_b*v_next_belief);
    }
    float tmp = m.get_expected_step_reward(b, i) + m.discount*poga_mult_bound;
    if( tmp > max_val)
    {
      aid = i;
      max_val = tmp;
    }
  }
  return max_val;
}

/*! calculates p(o | b,a) * V_low( new_belief(b, a, o) ) or
 *  p(o | b,a) * V_upper( new_belief(b, a, o) )
 *  depending upon is_lower
 */
float Solver::get_poga_mult_bound(Belief& b, int aid, int oid, float& lower_bound, float& upper_bound)
{
  Model& m = *model;
  //cout<<"aid: "<<aid<<" oid: "<<oid<<endl;

  Belief nb = m.next_belief(b, aid, oid);
  //nb.print();
  float poga = m.get_p_o_given_b(b, aid, oid);
  //cout<<"poga: "<<poga<<endl;

  lower_bound = poga*get_lower_bound_reward(nb);
  upper_bound = poga*get_mdp_upper_bound_reward(nb);

  return 0;
}

/*! samples beliefs
 * @param[in] : float epsilon difference between lower bound \
 * and upper bound at the root after backup is propagated upwards
 */
void Solver::sample(float epsilon)
{
  float L =  root_node->value_lower_bound;
  float U = L + epsilon;
  sample_beliefs(root_node, L, U, epsilon, 1);
}


/*! \return
 *  0: no dominance
 *  1: a1 dominates
 *  2: a2 dominates
 */
int Solver::check_alpha_dominated(Alpha& a1, Alpha& a2)
{
  int nnodes = belief_tree_nodes.size();
  int dominated_nodes = 0;
  for(unsigned int i=0; i< belief_tree_nodes.size(); i++)
  {
    BeliefNode* bn = belief_tree_nodes[i];
    if( a1.get_value(bn->b) >= a2.get_value(bn->b))
      dominated_nodes++;
    else
      dominated_nodes--;
  }
  if(dominated_nodes == nnodes)
    return 1;
  else if(dominated_nodes == -nnodes)
    return 2;
  else
    return 0;
}
/*! prunes alpha vectors O(n^2)
 * only_last flag checks only whether last vector can be removed
 * \return number of planes pruned
 */
int Solver::prune_alphas(bool only_last)
{
  if(only_last)
  {
    Alpha& alpha_last = alphas.back();
    int na = alphas.size()-2;
    for(int i=na; i>=0; i++)
    {
      int ret_val = check_alpha_dominated(alphas[i], alpha_last);
      cout<<"ret_val: "<<ret_val<<endl;
      if(ret_val == 1)
      {
        alphas.pop_back();
        return 1;
      }
      else if(ret_val == 2)
      {
        //cout<<"erased one inside"<<endl;
        alphas.erase(alphas.begin()+i);
        if(alphas.size() == 1)
          return na+1;
      }
    }
  }
  return 0;
}

/*! deletes all beliefnodes starting from root
 */
int Solver::trash_belief_tree(BeliefNode* root)
{
  return 0;
}

/*! prunes belief tree based on upper and lower bounds
 */
int Solver::prune_beliefs()
{
  for(auto i = belief_tree_nodes.rbegin(); i != belief_tree_nodes.rend(); i++)
  {
    BeliefNode* bn = *i;
    for(int j=0; j<m.nactions; j++)
    {
      int aid = j;
      float Ql = get_bound_child(bn->b, LOWER_BOUND, aid);
      float Qu = get_bound_child(bn->b, UPPER_BOUND, aid);
      if(Ql > Qu)
      {
        while(1)
        {
          // prune the whole subtree for children with actionid = aid
          auto it = find_if(bn->aoid.begin(), bn->aoid.end(), \
              [=]const pair<int, int> &p {return p.first == aid;});
          if(it != bn->aoid.end())
          {
            int index = it - bn->aoid.begin();
            BeliefNode* child = bn->children[index];
            // trash tree
            trash_belief_tree(child);
          }
          else
            break;
        }
      }
    }
  }
  return 0;
}

bool Solver::check_termination_condition(float ep)
{
  if( (root_node->value_upper_bound - root_node->value_lower_bound) > ep)
    return false;
  else
    return true;
}

void Solver::solve(float target_epsilon)
{
  float epsilon = 10;
  bool is_converged = false;
  int iteration = 0;
  cout<<"start solver"<<endl;
  while(!is_converged)
  {
    cout<<"iteration: "<< ++iteration << endl;
    sample(epsilon);

    cout<<"pruned: "<< prune_alphas(true) << endl;

    is_converged = check_termination_condition(target_epsilon);

    print_alphas();
    getchar();

    epsilon = epsilon/2.0;
  }
}
void Solver::sample_beliefs(BeliefNode* bn, float L, float U, float epsilon, int level)
{
  Model& m = *model;
  Belief& b = bn->b;

  float vhat = bn->value_prediction_optimal;
  float vupper = bn->value_upper_bound;
  float vlower = bn->value_lower_bound;
  cout<<"vupper: "<<vupper<<" vhat: "<<vhat<<" vlower: "<<vlower<<endl;

  cout<<"L: "<<L<<" U: "<<U<<" "<< "vlower + term: "<<vlower+ (float)0.5*epsilon*pow(m.discount, -level)<< endl;
  if( (vhat <= L) && (vupper < max(U, vlower+ (float)0.5*epsilon*pow(m.discount, -level))) )
  {
    cout<<"stop sampling: termination criterion reached"<<endl;
    return;
  }
  else
  {
    int tmp_aid = -1;
    float Qlower = get_bound_child(b, LOWER_BOUND, tmp_aid);
    float L1 = max(Qlower, L);
    float U1 = max(U, Qlower + pow(m.discount, -level)*epsilon);
    cout<<"Qlower: "<<Qlower <<" L1: "<<L1<<" U1: "<<U1<<endl;

    int new_action = -1;
    get_bound_child(b, UPPER_BOUND, new_action);
    //Belief new_belief = m.next_belief(b, new_action, -1);

    vector<float> poga_mult_lower_bounds(m.nobservations,0); 
    vector<float> poga_mult_upper_bounds(m.nobservations,0);
    int new_observation = -1;
    float max_diff_poga_mult_bound = -large_num;
    for(int i=0; i<m.nobservations; i++)
    {
      get_poga_mult_bound(b, new_action, i, poga_mult_lower_bounds[i], poga_mult_upper_bounds[i]);

      //cout<<"i: "<<i<<" poga bounds: "<< poga_mult_lower_bounds[i] <<" "<< poga_mult_upper_bounds[i]<<endl;
      if( (poga_mult_upper_bounds[i] - poga_mult_lower_bounds[i]) > max_diff_poga_mult_bound)
      {
        max_diff_poga_mult_bound = (poga_mult_upper_bounds[i] - poga_mult_lower_bounds[i]);
        new_observation = i;
      }
    }
    //cout<<"new_action: "<<new_action<<" new observation: "<< new_observation<< endl;
    float sum_tmp_lower = 0, sum_tmp_upper=0;
    for(int i=0; i< m.nobservations; i++)
    {
      if(i != new_observation)
        sum_tmp_lower = sum_tmp_lower + poga_mult_lower_bounds[i];
      if(i != new_observation)
        sum_tmp_upper = sum_tmp_upper + poga_mult_upper_bounds[i];
   }

    float expected_reward_new_action = m.get_expected_step_reward(b, new_action);
    float Lt = ((L1 - expected_reward_new_action)/m.discount - sum_tmp_lower)/m.get_p_o_given_b(b, new_action, new_observation);
    float Ut = ((U1 - expected_reward_new_action)/m.discount - sum_tmp_upper)/m.get_p_o_given_b(b, new_action, new_observation);

    Belief new_belief = m.next_belief(b, new_action, new_observation);

    BeliefNode* newbn = new BeliefNode(new_belief, bn, new_action, new_observation);
    belief_tree_nodes.push_back(newbn);
    insert_belief_node_into_tree(newbn);

    newbn->value_prediction_optimal = get_predicted_optimal_reward(newbn);
    newbn->value_upper_bound = get_mdp_upper_bound_reward(newbn->b);
    newbn->value_lower_bound = get_lower_bound_reward(newbn->b);

    newbn->print();
    cout<<"Lt: "<< Lt<<" Ut: "<<Ut << endl;
    //getchar();

    BeliefNode* curr = newbn;
    while(curr)
    {
      backup(newbn);
      curr = curr->parent;
    }
    sample_beliefs(newbn, Lt, Ut, epsilon, level+1);
  }
}
