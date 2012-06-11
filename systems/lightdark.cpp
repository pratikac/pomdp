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
        max_goal[i] = -0.6;
        
        min_controls[i] = -1.0;
        max_controls[i] = 1.0;
    
        min_right_beacon[i] = 0.8;
        max_right_beacon[i] = 2.0; 
    }
#if NUM_DIM==2
    init_state.x[0] = -1.0;
    init_state.x[1] = 1.5;
#elif NUM_DIM==1
    init_state.x[0] = 0;
#endif

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
    return t;
}

State System::integrate(State& s, double duration, bool is_clean)
{
    State t;
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
    State beacon; beacon.x[0] = 0.9; beacon.x[1] = 0.9;
    double dist2 = sq(beacon.x[0] - s.x[0]) + sq(beacon.x[1] - s.x[1]);
    for(int i=0; i<NUM_DIM_OBS; i++)
    {
        var[i] = obs_noise[i]*dist2 + 0.001;
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
            var[i] = 0.001;
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
    int how_many = 4;
    sampled_controls.clear();

    State ctmp; 
#if NUM_DIM==1
    ctmp.x[0] = max_controls[0]; sampled_controls.push_back(ctmp);
    ctmp.x[0] = min_controls[0]; sampled_controls.push_back(ctmp);
    ctmp.x[0] = 0; sampled_controls.push_back(ctmp);
    
    sampled_observations.clear();
    for(int i=0; i< how_many; i++)
    {
        State sobs;
        sobs.x[0] = min_states[0] + i/(float)how_many*(max_states[0] - min_states[0]);
        sampled_observations.push_back(sobs);
    }
#elif NUM_DIM==2
    ctmp.x[0] = max_controls[0]; ctmp.x[1] = 0;
    sampled_controls.push_back(ctmp);
    ctmp.x[0] = min_controls[0]; ctmp.x[1] = 0;
    sampled_controls.push_back(ctmp);
    ctmp.x[1] = max_controls[1]; ctmp.x[0] = 0;
    sampled_controls.push_back(ctmp);
    ctmp.x[1] = min_controls[1]; ctmp.x[0] = 0;
    sampled_controls.push_back(ctmp);
    ctmp.x[0] = 0; ctmp.x[1] = 0;
    sampled_controls.push_back(ctmp);
    
    sampled_observations.clear();
    for(int i=0; i< how_many; i++)
    {
        for(int j=0; j< how_many; j++)
        {
            State sobs;
            sobs.x[0] = min_states[0] + i/(float)how_many*(max_states[0] - min_states[0]);
            sobs.x[1] = min_states[1] + j/(float)how_many*(max_states[1] - min_states[1]);
            sampled_observations.push_back(sobs);
        }
    }
#endif


    return 0;
}

int System::get_lgq_path(double dT, vector<State>& lqg_path, vector<State>& lqg_covar, \
        vector<State>& lqg_control, double& total_cost)
{
    return 0;
}

