#include "lightdark.h"

System::System()
{
    name = "lightdark";

    min_states = new double[NUM_DIM];
    max_states = new double[NUM_DIM];
    
    min_goal = new double[NUM_DIM];
    max_goal = new double[NUM_DIM];
    
    min_left_beacon = new double[NUM_DIM];
    max_left_beacon = new double[NUM_DIM];
    min_right_beacon = new double[NUM_DIM];
    max_right_beacon = new double[NUM_DIM];

    max_controls = new double [NUM_DIM];
    min_controls = new double[NUM_DIM];
    
    obs_noise = new double[NUM_DIM];
    process_noise = new double[NUM_DIM];
    init_var = new double[NUM_DIM];

    for(int i=0; i< NUM_DIM; i++)
    {
        min_states[i] = -1;
        max_states[i] = 1;
        
        min_goal[i] = -1;
        max_goal[i] = -0.9;
        
        min_controls[i] = -0.5;
        max_controls[i] = 0.5;
    
        init_state.x[i] = 0;
    
        min_right_beacon[i] = 0.8;
        max_right_beacon[i] = 1.0; 
    }
   

    for(int i=0; i< NUM_DIM; i++)
    {
        process_noise[i] = 2*1e-2;
        obs_noise[i] = 1;
        init_var[i] = 1e-2;
    }
    sim_time_delta = 1e-3;
    discount = 0.95;

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
    
    delete[] min_left_beacon;
    delete[] max_left_beacon;
    delete[] min_right_beacon;
    delete[] max_right_beacon;

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
        t.x[i] = (control.x[i])*duration;
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

State System::integrate(State& s, double duration, bool is_clean)
{
    State t;

    double *var = new double[NUM_DIM];
    double *mean = new double[NUM_DIM];
    double *tmp = new double[NUM_DIM];

    for(int i=0; i<NUM_DIM; i++)
    {
        t.x[i] = s.x[i];
    }

    for(int i=0; i<NUM_DIM; i++)
    {   
        var[i] = process_noise[i]*( exp(duration) -1);
        tmp[i] = 0;
        mean[i] = 0;
    }
    if( !is_clean)  
        multivar_normal( mean, var, tmp, NUM_DIM);

    for(int i=0; i<NUM_DIM; i++)
        t.x[i] = exp(-duration)*t.x[i] + tmp[i];

    delete[] mean;
    delete[] tmp;
    delete[] var;

    return t;
}

State System::observation(State& s, bool is_clean)
{
    State t;

    double *tmp = new double[NUM_DIM_OBS];
    double *mean = new double[NUM_DIM_OBS];

    if( !is_clean)  
        multivar_normal( mean, obs_noise, tmp, NUM_DIM_OBS);
    else
    {
        for(int i=0; i<NUM_DIM_OBS; i++)
            tmp[i] = 0;
    }

    //double range = s.norm();
    //double theta = atan2(s.x[1], s.x[0]);
    
    for(int i=0; i<NUM_DIM_OBS; i++)
        t.x[i] = s.x[i] + tmp[i];
    
    delete[] mean;
    delete[] tmp;

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
#if 0
    for(int i=0; i<NUM_DIM_OBS; i++)
    {
        var[i] = obs_noise[i]*pow((0.9 - s.x[i]), 2) + 1e-4;
    }
#else
    bool flag = false;
    if (NUM_DIM==1)
        flag = (s.x[0] >= min_right_beacon[0]) && (s.x[0] <= max_right_beacon[0]);
    else if(NUM_DIM==2)
        flag = (s.x[0] >= min_right_beacon[0]) && (s.x[0] <= max_right_beacon[0])
          &&  (s.x[1] >= min_right_beacon[1]) && (s.x[1] <= max_right_beacon[1]);
    
    if(flag)
    {
        for(int i=0; i<NUM_DIM_OBS; i++)
            var[i] = obs_noise[i]/1000;
    }
    else
    {
        for(int i=0; i<NUM_DIM_OBS; i++)
            var[i] = obs_noise[i];
    }
#endif
}

int System::get_lgq_path(double dT, vector<State>& lqg_path, vector<State>& lqg_covar, \
        vector<State>& lqg_control, double& total_cost)
{
    return 0;
}

