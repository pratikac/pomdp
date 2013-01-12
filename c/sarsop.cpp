#include "sarsop.h"

using namespace sarsop;

/*! basic value iteration for MDP
 * initializes upper bound of value function
 */
void Solver::mdp_value_iteration()
{
  Model& m = *model;

  vec mdp_value_copy(mdp_value);
  double epsilon = 1e-2;
  bool is_converged = false;
  while(!is_converged)
  {
    mdp_value_copy = mdp_value;

    is_converged = true;
    for(int i=0; i< m.nstates; i++)
    {
      double max_value = -large_num;
      for(int j=0; j< m.nactions; j++)
      {
        double tmp = 0;
        for(int k=0; k< m.nstates; k++)
        {
          tmp = tmp + m.ptransition[j][k][i]*m.discount*mdp_value_copy[k];
        }
        tmp = tmp + m.preward[j][i]; 
        if (tmp > max_value)
          max_value = tmp;
      }
      if( fabs(max_value - mdp_value_copy[i]) > epsilon)
        is_converged = false;
      mdp_value[i] = max_value;
    }
    //print_vec(mdp_value);
  }
  //cout<<"mdp_value: ";
  //print_vec(mdp_value);
}

/*! calculates one single alpha plane for the best fixed action policy (HSVI2 paper)
*/
void Solver::fixed_action_alpha_iteration()
{
  Model& m = *model;

  Alpha alpha;
  alpha.gradient = vec(m.nstates,0);

  int fixed_aid = -1;
  double min_reward = large_num;
  for(int j=0; j< m.nactions; j++)
  {
    for(int i=0; i < m.nstates; i++)
    {
      if(min_reward > m.preward[j][i])
      {
        min_reward = m.preward[j][i];
        fixed_aid = j;
      }
    }
  }
  alpha.actionid = fixed_aid;
  for(int i=0; i< m.nstates; i++)
    alpha.gradient[i] = min_reward/(1.0 - m.discount);

  alphas.push_back(alpha);
  print_alphas();
  /*
     fixed_action_alphas = vector<Alpha>(m.nactions);
     for(int i=0; i< m.nactions; i++)
     {
     fixed_action_alphas[i].actionid = i;
     fixed_action_alphas[i].gradient = vec(m.nstates, 0);
     }

     vector<Alpha> fixed_action_alphas_copy;
     for(int c=0; c<10; c++)
     {
     fixed_action_alphas_copy = fixed_action_alphas;
     for(int i=0; i< m.nactions; i++)
     {
     for(int j=0; j< m.nstates; j++)
     {
     for(int k=0; k< m.nstates; k++)
     {
     fixed_action_alphas[i].gradient[j] += (m.prewardp[i][j] + m.discount*
     m.ptransition[i][k][j]*fixed_action_alphas_copy[i].gradient[k]);
     }
     }
     }
     }
     prune(fixed_action_alphas);
     */
  //cout<<"fixed_action_initialize: "<<endl;
  //print_alphas();
}

void Solver::backup(BeliefNode* bn)
{
  Model& m = *model;
  int na = m.nactions;
  int no = m.nobservations;
  int ns = m.nstates;

  BeliefNode* curr = bn;
  while(curr->parent != NULL)
  {
    Belief b = curr->b;
    matrix_i alpha_ao = matrix_i(na, vec_i(no));

    for(int i=0; i<na; i++)
    {
      for(int j=0; j<no; j++)
      {
        int max_alpha_id = -1;
        double max_val = -large_num;
        for(unsigned int k=0; k < alphas.size(); k++)
        {
          Belief tmp_belief = m.next_belief(b, i, j);
          double tmp = alphas[k].get_value(tmp_belief);
          if(tmp > max_val)
          {
            max_val = tmp;
            max_alpha_id = k;
          }
        }
        alpha_ao[i][j] = max_alpha_id;
      }
    }
    //print_mat(alpha_id);
    //getchar();

    Alpha new_alpha, tmp_alpha;
    tmp_alpha.gradient = vec(m.nstates, 0);
    double max_val = -large_num;
    for(int i=0; i<na; i++)
    {
      tmp_alpha.actionid = i;
      for(int j=0; j<ns; j++)
      {
        double tmp = 0;
        for(int k=0; k<ns; k++)
        {
          for(int l=0; l<no; l++)
          {
            tmp = tmp + (m.ptransition[i][k][j]*m.pobservation[i][l][k]*(alphas[alpha_ao[i][l]].gradient[k]));
          }
        }
        tmp_alpha.gradient[j] = m.preward[i][j] + m.discount*tmp;
      }
      double tmp_alpha_val = tmp_alpha.get_value(b);
      if( tmp_alpha_val > max_val)
        new_alpha = tmp_alpha;
    }

    alphas.push_back(new_alpha);
    prune(true);

    curr = curr->parent;
  }
}

/*! predict optimal reward using binning of beliefs
*/
double Solver::get_predicted_optimal_reward(Belief& b)
{
  return get_lower_bound_reward(b);
}

/*! lower bound using alpha vectors
*/
double Solver::get_lower_bound_reward(Belief& b)
{
  double max_value = -large_num;
  for(unsigned int i=0; i< alphas.size(); i++)
  {
    double tmp = alphas[i].get_value(b);
    if(tmp > max_value)
      max_value = tmp;
  }
  return max_value;
}

/*! upper bound from the mdp
*/
double Solver::get_upper_bound_reward(Belief& b)
{
  return dot(b.p, mdp_value);
  /*
     double tmp = 0;
     for(int i=0; i< model->nstates; i++)
     {
     tmp = tmp + b.p[i]*mdp_value[i];
     }
     return tmp;
     */
}

double Solver::get_bound_child(Belief& b, bool is_lower, int& aid)
{
  Model& m = *model;

  double max_val = -large_num;
  for(int i=0; i< m.nactions; i++)
  {
    double poga_mult_bound = 0;
    for(int j=0; j< m.nobservations; j++)
    {
      double v_next_belief = 0;
      Belief new_belief_tmp = m.next_belief(b, i, j);
      if(is_lower == true)
        v_next_belief = get_lower_bound_reward(new_belief_tmp);
      else
        v_next_belief = get_upper_bound_reward(new_belief_tmp);

      double p_o_given_b = m.get_p_o_given_b(b, i, j);
      poga_mult_bound = poga_mult_bound + (p_o_given_b*v_next_belief);
    }
    double tmp = m.get_expected_step_reward(b, i) + m.discount*poga_mult_bound;
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
double Solver::get_poga_mult_bound(Belief& b, int aid, int oid, double& lower_bound, double& upper_bound)
{
  Model& m = *model;
  //cout<<"aid: "<<aid<<" oid: "<<oid<<endl;

  Belief nb = m.next_belief(b, aid, oid);
  //nb.print();
  double poga = m.get_p_o_given_b(b, aid, oid);
  //cout<<"poga: "<<poga<<endl;

  lower_bound = poga*get_lower_bound_reward(nb);
  upper_bound = poga*get_upper_bound_reward(nb);

  return 0;
}

void Solver::sample(double epsilon)
{
  double L =  root_node->value_lower_bound;
  double U = L + epsilon;

  sample_beliefs(root_node, L, U, epsilon, 1);
}

#define lower   true
#define upper   false
void Solver::sample_beliefs(BeliefNode* bn, double L, double U, double epsilon, int level)
{
  Model& m = *model;
  Belief& b = bn->b;

  double vhat = bn->value_prediction_optimal;
  double vupper = bn->value_upper_bound;
  double vlower = bn->value_lower_bound;
  //cout<<"vupper: "<<vupper<<" vhat: "<<vhat<<" vlower: "<<vlower<<endl;

  //cout<<"L: "<<L<<" U: "<<U<<" "<< "vlower + term: "<<vlower+ (double)0.5*epsilon*pow(m.discount, -level)<< endl;
  if( (vhat <= L) && (vupper < max(U, vlower+ (double)0.5*epsilon*pow(m.discount, -level))) )
  {
    //cout<<"termination criterion reached"<<endl;
    return;
  }
  else
  {
    int tmp_aid = -1;
    double Qlower = get_bound_child(b, lower, tmp_aid);
    double L1 = max(Qlower, L);
    double U1 = max(U, Qlower + pow(m.discount, -level)*epsilon);
    //cout<<"Qlower: "<<Qlower <<" L1: "<<L1<<" U1: "<<U1<<endl;

    int new_action = -1;
    get_bound_child(b, upper, new_action);
    //Belief new_belief = m.next_belief(b, new_action, -1);

    vector<double> poga_mult_lower_bounds(m.nobservations,0); 
    vector<double> poga_mult_upper_bounds(m.nobservations,0);
    int new_observation = -1;
    double max_diff_poga_mult_bound = -large_num;
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
    double sum_tmp_lower = 0, sum_tmp_upper=0;
    for(int i=0; i< m.nobservations; i++)
    {
      if(i != new_observation)
        sum_tmp_lower = sum_tmp_lower + poga_mult_lower_bounds[i];
      if(i != new_observation)
        sum_tmp_upper = sum_tmp_upper + poga_mult_upper_bounds[i];
    }

    double expected_reward_new_action = m.get_expected_step_reward(b, new_action);
    double Lt = ((L1 - expected_reward_new_action)/m.discount - sum_tmp_lower)/m.get_p_o_given_b(b, new_action, new_observation);
    double Ut = ((U1 - expected_reward_new_action)/m.discount - sum_tmp_upper)/m.get_p_o_given_b(b, new_action, new_observation);

    Belief new_belief = m.next_belief(b, new_action, new_observation);

    BeliefNode* newbn = new BeliefNode(new_belief, bn, new_action, new_observation);
    belief_tree_nodes.push_back(newbn);
    insert_belief_node_into_tree(newbn);

    newbn->value_prediction_optimal = get_predicted_optimal_reward(newbn->b);
    newbn->value_upper_bound = get_upper_bound_reward(newbn->b);
    newbn->value_lower_bound = get_lower_bound_reward(newbn->b);

    //newbn->print();
    //cout<<"Lt: "<< Lt<<" Ut: "<<Ut << endl;
    //getchar();

    backup(newbn);

    sample_beliefs(newbn, Lt, Ut, epsilon, level+1);
  }
}


/*! returns
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
    if( a1.get_value(bn->b) > a2.get_value(bn->b))
      dominated_nodes++;
    else
      dominated_nodes--;
  }
  if(dominated_nodes == nnodes)
    return 1;
  else if(dominated_nodes == -nnodes)
    return 2;

  return 0;
}
/*! prunes alpha vectors O(n^2)
 * only_last flag checks only whether last vector can be removed
 * \return number of planes pruned
 */
int Solver::prune(bool only_last)
{
  if(only_last)
  {
    Alpha& alpha_last = alphas.back();
    int na = alphas.size()-2;
    for(int i=na; i>=0; i++)
    {
      int ret_val = check_alpha_dominated(alphas[i], alpha_last);
      //cout<<"ret_val: "<<ret_val<<endl;
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

bool Solver::check_termination_condition(double ep)
{
  if( (root_node->value_upper_bound - root_node->value_lower_bound) > ep)
    return false;
  else
    return true;
}

void Solver::initialize()
{
  mdp_value_iteration();
  fixed_action_alpha_iteration();

  root_node->value_lower_bound = get_lower_bound_reward(root_node->b);
  root_node->value_upper_bound = get_upper_bound_reward(root_node->b);
  root_node->value_prediction_optimal = get_predicted_optimal_reward(root_node->b);

  cout<<"root_node: "<<endl; root_node->print(); cout<<endl;
}

void Solver::solve(double target_epsilon)
{
  double epsilon = 10;
  bool is_converged = false;
  int iteration = 0;
  cout<<"start solver"<<endl;
  while(!is_converged)
  {
    cout<<"iteration: "<< ++iteration << endl;
    sample(epsilon);

    //int how_many = prune(false);

    is_converged = check_termination_condition(target_epsilon);

    print_alphas();
    //getchar();

    epsilon = epsilon/2.0;
  }
}
