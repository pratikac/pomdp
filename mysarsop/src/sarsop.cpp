#include "sarsop.h"

using namespace sarsop;

/*! basic value iteration for MDP
 * initializes upper bound of value function
 */
void Solver::mdp_value_iteration()
{
    Model& m = *model;

    vec mdp_value_copy(mdp_value);
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
                float tmp = 0;
                for(int k=0; k< m.nstates; k++)
                {
                    tmp = tmp + m.ptransition[j][i][k]*(m.preward[i][j] + m.discount*mdp_value_copy[k]);
                }
                if (tmp > max_value)
                    max_value = tmp;
            }
            if( fabs(max_value - mdp_value_copy[i]) > epsilon)
                is_converged = false;
            mdp_value[i] = max_value;
        }
        //print_vec(mdp_value);
    }
    print_vec(mdp_value);
}

void Solver::fixed_action_alpha_iteration()
{
    Model& m = *model;
    
    Alpha alpha;
    alpha.gradient = vec(m.nstates);

    float max_lower_bound = -large_num;
    int fixed_aid = -1;
    for(int j=0; j< m.nactions; j++)
    {
        float min_reward = large_num;
        for(int i=0; i < m.nstates; i++)
        {
            if(min_reward > m.preward[i][j])
                min_reward = m.preward[i][j];
        }
        if(max_lower_bound < min_reward)
        {
            fixed_aid = j;
            max_lower_bound = min_reward;
    }
    }
    alpha.actionid = fixed_aid;
    for(int i=0; i< m.nstates; i++)
        alpha.gradient[i] = max_lower_bound/(1- m.discount);

    fixed_action_alphas.push_back(alpha);

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
                    fixed_action_alphas[i].gradient[j] += (m.preward[j][i] + m.discount*
                            m.ptransition[i][j][k]*fixed_action_alphas_copy[i].gradient[k]);
                }
            }
        }
    }
    prune(fixed_action_alphas);
    */
    for(unsigned int i=0; i<fixed_action_alphas.size(); i++)
    {
        cout<<fixed_action_alphas[i].actionid<<": ";
        print_vec(fixed_action_alphas[i].gradient);
    }
}

void Solver::backup(Belief& b)
{
    Model& m = *model;
    int na = m.nactions;
    int no = m.nobservations;
    int ns = m.nstates;

    matrix_i alpha_id = matrix_i(na, vec_i(no));

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
            alpha_id[i][j] = max_alpha_id;
        }
    }

    Alpha new_alpha, tmp_alpha;
    float max_val = -large_num;
    for(int i=0; i<na; i++)
    {
        tmp_alpha.actionid = i;
        for(int j=0; j<ns; j++)
        {
            float tmp = 0;
            for(int k=0; k<ns; k++)
            {
                for(int l=0; l<no; l++)
                {
                    tmp += m.ptransition[i][j][k]*m.pobservation[l][k]*alphas[alpha_id[i][l]].gradient[k];
                }
            }
            tmp_alpha.gradient[j] = m.preward[j][i] + m.discount*tmp;
        }
        float tmp_alpha_val = tmp_alpha.get_value(b);
        if( tmp_alpha_val > max_val)
            new_alpha = tmp_alpha;
    }

    alphas.push_back(new_alpha);
}

/*! predict optimal reward using binning of beliefs
 */
float Solver::predicted_optimal_reward(Belief& b)
{
    return 0;
}

/*! lower bound using some constant action policy
 */
float Solver::lower_bound_reward(Belief& b)
{
    float max_value = -large_num;
    for(int i=0; i< model->nactions; i++)
    {
        float tmp = fixed_action_alphas[i].get_value(b);
        if(tmp > max_value)
            max_value = tmp;
    }
    return max_value;
}

/*! upper bound from the mdp
 */
float Solver::upper_bound_reward(Belief& b)
{
    float tmp = 0;
    for(int i=0; i< model->nstates; i++)
    {
        tmp += b.p[i]*mdp_value[i];
    }
    return tmp;
}

float Solver::get_bound_child(Belief& b, bool is_lower, int& aid)
{
    Model& m = *model;

    float max_val = -large_num;
    for(int i=0; i< m.nactions; i++)
    {
        float poga_mult_bound = 0;
        for(int j=0; j< m.nobservations; j++)
        {
            float v_next_belief = -large_num;
            Belief new_belief_tmp = m.next_belief(b, i, j);
            if(is_lower == true)
                v_next_belief = lower_bound_reward(new_belief_tmp);
            else
                v_next_belief = upper_bound_reward(new_belief_tmp);

            float p_o_given_b = 0;
            for(int k=0; k< m.nstates; k++)
            {
                p_o_given_b += m.pobservation[j][k]*(b.p[k]);
            }
            poga_mult_bound = p_o_given_b*v_next_belief;
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

float Solver::get_poga_mult_bound(Belief& b, int aid, int oid, bool is_lower)
{
    return 0;
}

void Solver::sample(float epsilon)
{
    float L =  root_node->value_lower_bound;
    float U = L + epsilon;

    sample_belief_points(root_node, L, U, epsilon, 1);
}

#define lower   true
#define upper   false
void Solver::sample_belief_points(BeliefNode* bn, float L, float U, float epsilon, int level)
{
    Model& m = *model;
    Belief b = bn->b;

    float vhat = bn->value_prediction_optimal;
    float vupper = bn->value_upper_bound;
    float vlower = bn->value_lower_bound;

    if( (vhat <= L) && (vupper <= max(U, vlower+ (float)2.0*epsilon*pow(m.discount, -level)) ) )
        return;
    else
    {
        int tmp_aid;
        float Qlower = get_bound_child(bn->b, lower, tmp_aid);
        float L1 = max(Qlower, L);
        float U1 = max(U, Qlower + pow(m.discount, -level)*epsilon);
        
        int new_action = -1;
        get_bound_child(bn->b, upper, new_action);
        Belief new_belief = m.next_belief(b, new_action, -1);
        
        vector<float> poga_mult_lower_bounds(m.nobservations); 
        vector<float> poga_mult_upper_bounds(m.nobservations);
        int new_observation = -1;
        float max_diff_poga_mult_bound = -large_num;
        for(int i=0; i<m.nobservations; i++)
        {
            poga_mult_upper_bounds[i] = get_poga_mult_bound(new_belief, new_action, i, upper);
            poga_mult_lower_bounds[i] = get_poga_mult_bound(new_belief, new_action, i, lower);
            
            if( (poga_mult_upper_bounds[i] - poga_mult_lower_bounds[i]) > max_diff_poga_mult_bound)
            {
                max_diff_poga_mult_bound = (poga_mult_upper_bounds[i] - poga_mult_lower_bounds[i]);
                new_observation = i;
            }
        }
        float sum_tmp_lower = 0, sum_tmp_upper=0;
        for(int i=0; i< m.nobservations; i++)
        {
            if(i != new_observation)
                sum_tmp_lower += poga_mult_lower_bounds[i];
            if(i != new_observation)
                sum_tmp_upper += poga_mult_upper_bounds[i];
        }

        float expected_reward_new_action = m.get_expected_step_reward(b, new_action);
        float Lt = (L1 - expected_reward_new_action)/m.discount - m.get_p_o_given_b(b, new_observation) - sum_tmp_lower;
        float Ut = (U1 - expected_reward_new_action)/m.discount - m.get_p_o_given_b(b, new_observation) - sum_tmp_upper;

        new_belief = m.next_belief(b, new_action, new_observation);

        BeliefNode* newbn = new BeliefNode(new_belief, bn, new_action, new_observation);
        belief_tree_nodes.push_back(newbn);
        insert_belief_node_into_tree(newbn);
        
        newbn->value_prediction_optimal = predicted_optimal_reward(new_belief);
        newbn->value_upper_bound = upper_bound_reward(new_belief);
        newbn->value_lower_bound = lower_bound_reward(new_belief);
        
        backup(b);

        sample_belief_points(newbn, Lt, Ut, epsilon, level+1);
    }
}


int Solver::prune(vector<Alpha>& avec)
{
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
    mdp_value_iteration();
    fixed_action_alpha_iteration();
    
    float epsilon = 100;
    bool is_converged = false;
    while(!is_converged)
    {
        sample(epsilon);

        int how_many = prune(alphas);
        
        is_converged = check_termination_condition(target_epsilon);
        
        if(how_many > 5)
            epsilon = epsilon/2.0;
    }
}
