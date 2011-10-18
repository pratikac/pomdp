#include "singleint.h"

System::System()
{
    min_states = new double[NUM_DIM];
    max_states = new double[NUM_DIM];
    max_controls = new double [NUM_DIM];
    min_controls = new double[NUM_DIM];

    obs_noise = new double[NUM_DIM];
    process_noise = new double[NUM_DIM];
    init_var = new double[NUM_DIM];

    for(int i=0; i< NUM_DIM; i++)
    {
        min_states[i] = -1;
        max_states[i] = 1.5;
        init_state.x[i] = 1;
    }
    
    for(int i=0; i< NUM_DIM; i++)
    {
        process_noise[i] = 1e-3;
        obs_noise[i] = 1e-3;
        init_var[i] = 1e-3;
    }
    sim_time_delta = 1e-3;
}

System::~System()
{
    delete[] min_states;
    delete[] max_states;
    delete[] min_controls;
    delete[] max_controls;

    delete[] obs_noise;
    delete[] process_noise;
    delete[] init_var;
}

State System::get_fdt(State& s, double duration)
{
    State t;

    for(int i=0; i<NUM_DIM; i++)
    {
        t.x[i] = s.x[i];
    }

    for(int i=0; i<NUM_DIM; i++)
        t.x[i] = -t.x[i]*duration;
    
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



void System::get_variance(State& s, double duration, double* var)
{
    for(int i=0; i<NUM_DIM; i++)
    {   
        var[i] = process_noise[i]*( exp(duration) -1);
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

