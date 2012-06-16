#include "sarsop.h"

using namespace sarsop;

void Solver::backup(Belief* b)
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
                Belief tmp_belief = m.next_belief(*b, i, j);
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
        float tmp_alpha_val = tmp_alpha.get_value(*b);
        if( tmp_alpha_val > max_val)
            new_alpha = tmp_alpha;
    }

    alphas.push_back(new_alpha);
}

float Solver::predicted_optimal_reward(Belief& b)
{
    return 0;
}

float Solver::lower_bound_reward(Belief& b)
{
    return 0;
}

float Solver::upper_bound_reward(Belief& b)
{
    return 0;
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

void Solver::sample(float target_epsilon)
{
    float L =  root_node->lower_bound;
    float U = L + target_epsilon;

    sample_belief_points(root_node, L, U, target_epsilon, 1);
}

#define lower   true
#define upper   false
void Solver::sample_belief_points(BeliefNode* bn, float L, float U, float epsilon, int level)
{
    Model& m = *model;
    Belief b = bn->b;

    float vhat = predicted_optimal_reward(b);
    float vupper = upper_bound_reward(b);
    float vlower = lower_bound_reward(b);

    if( (vhat <= L) && (vupper <= max(U, vlower+epsilon*pow(m.discount, -level)) ) )
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

        sample_belief_points(newbn, Lt, Ut, epsilon, level+1);
    }
}
