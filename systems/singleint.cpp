#include "singleint.h"

System::System()
{
    min_states = new double[NUM_DIM];
    max_states = new double[NUM_DIM];
    
    min_goal = new double[NUM_DIM];
    max_goal = new double[NUM_DIM];

    max_controls = new double [NUM_DIM];
    min_controls = new double[NUM_DIM];
    
    obs_noise = new double[NUM_DIM];
    process_noise = new double[NUM_DIM];
    init_var = new double[NUM_DIM];

    for(int i=0; i< NUM_DIM; i++)
    {
        min_states[i] = 0;
        max_states[i] = 0.5;
        
        min_goal[i] = 0;
        max_goal[i] = 0.05;
        
        init_state.x[i] = 0.4;

        min_controls[i] = 0;
        max_controls[i] = 0.5;
    }
    
    for(int i=0; i< NUM_DIM; i++)
    {
        process_noise[i] = 1e-2;
        obs_noise[i] = 1e-2;
        init_var[i] = 1e-3;
    }
    sim_time_delta = 1e-3;
    
    controls_tree = kd_create(NUM_DIM);
    // sample controls, add zero control to make any region as goal region
    for(int i=0; i< 10; i++)
    {
        State ctmp = sample_control();
        sampled_controls.push_back(ctmp);
        kd_insert(controls_tree, ctmp.x, &(sampled_controls.back()));
    }

    State ctmp;
    for(int i=0; i<NUM_DIM; i++)
        ctmp.x[i] = 0;

    sampled_controls.push_back(ctmp);
    kd_insert(controls_tree, ctmp.x, &(sampled_controls.back()));
}

System::~System()
{   
    delete[] min_states;
    delete[] max_states;

    delete[] min_goal;
    delete[] max_goal;

    delete[] min_controls;
    delete[] max_controls;

    delete[] obs_noise;
    delete[] process_noise;
    delete[] init_var;

    kd_free(controls_tree);
}

State System::get_fdt(State& s, State& control, double duration)
{
    State t;
    for(int i=0; i< NUM_DIM; i++)
    {
        t.x[i] = (-s.x[i] + control.x[i])*duration;
        //cout << t.x[i] << endl;
    }
    return t;
}

State System::get_controller(State& s)
{
    State t;
    for(int i=0; i< NUM_DIM; i++)
        t.x[i] = -5*s.x[i];

    return t;
}

State System::integrate(State& s, State& control, double duration, bool is_clean)
{
    State t;

    double *var = new double[NUM_DIM];
    double *mean = new double[NUM_DIM];
    double *tmp = new double[NUM_DIM];

    double delta_t = 1e-2;
    for(int i=0; i<NUM_DIM; i++)
    {   
        var[i] = process_noise[i]*delta_t;
        tmp[i] = 0;
        mean[i] = 0;
    }
    
    double curr_time = 0;
    while(curr_time < duration)
    {
        if( !is_clean)  
            multivar_normal( mean, var, tmp, NUM_DIM);

        for(int i=0; i<NUM_DIM; i++)
            t.x[i] = ( -t.x[i] + control.x[i])*delta_t + tmp[i];

        curr_time += delta_t;
    }

    delete[] mean;
    delete[] tmp;
    delete[] var;

    return t;
}

State System::integrate_alpha(double alpha, State& s, State& control, double duration, bool is_clean)
{
    State t = s;

    double *var = new double[NUM_DIM];
    double *mean = new double[NUM_DIM];
    double *tmp = new double[NUM_DIM];

    double delta_t = 1e-2;
    for(int i=0; i<NUM_DIM; i++)
    {   
        var[i] = process_noise[i]*delta_t;
        tmp[i] = 0;
        mean[i] = 0;
    }

    double curr_time = 0;
    while(curr_time < duration)
    {
        if( !is_clean)  
            multivar_normal( mean, var, tmp, NUM_DIM);

        for(int i=0; i<NUM_DIM; i++)
            t.x[i] = ( (alpha - 1)*t.x[i] + control.x[i])*delta_t  + t.x[i] + tmp[i];

        curr_time += delta_t;
    }
    delete[] mean;
    delete[] tmp;
    delete[] var;

    return t;
}

void System::get_variance(State& s, double duration, double* var)
{
    for(int i=0; i<NUM_DIM; i++)
    {   
        var[i] = process_noise[i]*duration;
    } 
}
void System::get_obs_variance(State& s, double* var)
{
    for(int i=0; i<NUM_DIM_OBS; i++)
    {   
        var[i] = obs_noise[i];
    } 
}

State System::observation(State& s, bool is_clean)
{
    State t;

    double *tmp = new double[NUM_DIM];
    double *mean = new double[NUM_DIM];

    if( !is_clean)  
        multivar_normal( mean, obs_noise, tmp, NUM_DIM);
    else
    {
        for(int i=0; i<NUM_DIM; i++)
            tmp[i] = 0;
    }

    for(int i=0; i<NUM_DIM; i++)
        t.x[i] = s.x[i] + tmp[i-1];

    delete[] mean;
    delete[] tmp;

    return t;
}

int System::get_lgq_path(double dT, vector<State>& lqg_path, vector<State>& lqg_covar, \
        vector<State>& lqg_control, double& total_cost)
{
    lqg_path.clear();
    lqg_covar.clear();
    lqg_control.clear();

    int traj_len = 100;
    double discount = 0.99;
    double alpha = (1-discount)/2/dT;

    lqg_path.push_back(init_state);
    
    State curr_state;
    multivar_normal(init_state.x, init_var, curr_state.x, NUM_DIM);
   
    double *K = new double[traj_len];
    double P = 0;
    // solve for controller K offline, finite time discrete LQR
    for(int i=traj_len-1; i >= 0; i--)
    {   
        K[i] = 1/(1 + P)*P*(1+dT*(1+alpha));
        P = 1 + K[i]*K[i] + (1+dT*(1+alpha) - K[i])*(1+dT*(1+alpha) -K[i])*P;
    }
    
    // initialize Kalman filter
    State curr_xhat = init_state;
    State Q;
    for(int j=0; j<NUM_DIM; j++)
        Q.x[j] = init_var[j];

    total_cost = 0;
    for(int i=0; i< traj_len; i++)
    {
        State curr_control;
        for(int j=0; j<NUM_DIM;j++)
            curr_control.x[j] = -K[i]*curr_xhat.x[j];

        State next_state = integrate_alpha(alpha, curr_state, curr_control, dT, false);
        State next_xhat = integrate_alpha(alpha, curr_xhat, curr_control, dT, true);
        
        State next_obs = observation(next_state, false);
        State clean_obs = observation(next_state, true);

        for(int j=0; j < NUM_DIM; j++)
        {
            Q.x[j] = Q.x[j]*pow( (alpha-1-K[i])*dT + 1, 2) + process_noise[j]*dT;
            double S = next_obs.x[j] - clean_obs.x[j];
            double L = Q.x[j]/(Q.x[j] + obs_noise[j]);

            next_xhat.x[j] += L*S;

            Q.x[j] = (1 -L)*Q.x[j];
        }

        total_cost += (0.5*exp(2*alpha*i*dT)*(next_state.norm2() + curr_control.norm2()));
        curr_xhat = next_xhat;
        curr_state = next_state;

        lqg_path.push_back(curr_state);

        lqg_covar.push_back(Q);
        lqg_control.push_back(curr_control);
    }
    
    delete [] K;

    return 0;
}
