#include "lightdark.h"

System::System(double discount_factor, double process_noise_in)
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
        min_states[i] = -2;
        max_states[i] = 2;
        
        min_goal[i] = -1.0;
        max_goal[i] = -0.8;
        
        min_controls[i] = -1.0;
        max_controls[i] = 1.0;
    
        init_state.x[i] = 0;
    
        min_right_beacon[i] = 0.8;
        max_right_beacon[i] = 1.0; 
    }
   

    for(int i=0; i< NUM_DIM; i++)
    {
        process_noise[i] = process_noise_in;
        obs_noise[i] = 10;
        init_var[i] = 1e-1;
    }
    sim_time_delta = 1e-3;
    discount = discount_factor;
    
    //cout<<"system_init: "<< discount<<" "<< process_noise[0] << endl;
    controls_tree = kd_create(NUM_DIM);
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
        t.x[i] = -0.1*s.x[i];

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
        var[i] = process_noise[i]*duration;
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
        var[i] = obs_noise[i]*pow((0.9 - s.x[i]), 2) + 0.01;
    }
    return;
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
            var[i] = 0.01;
    }
    else
    {
        for(int i=0; i<NUM_DIM_OBS; i++)
            var[i] = 100;
    }
#endif
}

int System::sample_control_observations(int num_vert)
{
    int how_many = 10; // 3*log(num_vert);
    sampled_controls.clear();
#if 0
    cout<<"sampling: "<< how_many <<" controls and observations"<<endl;
    // sample controls, add zero control to make any region as goal region
    for(int i=0; i< how_many-1; i++)
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
    // cout<<"sampled controls: " << sampled_controls.size() << endl;
#else
    State ctmp; 
    ctmp.x[0] = max_controls[0]; sampled_controls.push_back(ctmp);
    ctmp.x[0] = min_controls[0]; sampled_controls.push_back(ctmp);
    //ctmp.x[0] = 0; sampled_controls.push_back(ctmp);
#endif
    sampled_observations.clear();
    for(int i=0; i< how_many; i++)
    {
        State sobs;
        for(int j=0; j< NUM_DIM; j++)
            sobs.x[j] = min_states[j] + i/(float)how_many*(max_states[j] - min_states[j]);
        sampled_observations.push_back(sobs);
        //sobs.print();
    }

    return 0;
}

int System::get_lgq_path(double dT, vector<State>& lqg_path, vector<State>& lqg_covar, \
        vector<State>& lqg_control, double& total_cost)
{
    return 0;
}

