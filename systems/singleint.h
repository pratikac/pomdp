#ifndef __singleint_h__
#define __singleint_h__

#include "../utils/common.h"
#define NUM_DIM         (1)
#define NUM_DIM_OBS     (1)
// no time in this algorithm

class State
{
    public:
        double x[NUM_DIM];

        State()
        {
            for(int i=0; i<NUM_DIM; i++)
                x[i] = 0;
        }
        State(double *val)
        {
            for(int i=0; i<NUM_DIM; i++)
                x[i] = val[i];
        }
        ~State()
        {
        }

        double operator[](int which_dim)
        {
            assert(which_dim < NUM_DIM);
            return x[which_dim];
        }
        
        double norm2()
        {
            double sum = 0;
            for(int i=0; i< NUM_DIM; i++)
                sum += (x[i]*x[i]);

            return sum;
        }
        double norm()
        {
            double sum = 0;
            for(int i=0; i< NUM_DIM; i++)
                sum += (x[i]*x[i]);

            return sqrt(sum);
        }


        double operator*(const State& s1)
        {
            double ret = 0;
            for(int i=0; i< NUM_DIM; i++)
                ret += (this->x[i]*s1.x[i]);

            return ret;
        }
        State operator+(const State& s1)
        {
            State ret;
            for(int i=0; i< NUM_DIM; i++)
                ret.x[i] = (this->x[i] + s1.x[i]);

            return ret;
        }
        State operator-(const State& s1)
        {
            State ret;
            for(int i=0; i< NUM_DIM; i++)
                ret.x[i] = (this->x[i] - s1.x[i]);

            return ret;
        }
        State& operator=(const State &that)
        {
            if(this != &that)
            {
                for(int i=0; i< NUM_DIM; i++)
                    x[i] = that.x[i];
                
                return *this;
            }
            else
                return *this;
        }
        void print(ofstream &which_stream)
        {
            for(int i=0; i< NUM_DIM; i++)
                which_stream<< x[i] << "\t";

            which_stream<< endl;
        }
};

class System
{
    public:
        
        string name;

        double *obs_noise;
        double *process_noise;
        double *init_var;

        double *min_states;
        double *max_states;
        
        double *min_goal;
        double *max_goal;

        double *min_controls;
        double *max_controls;
         
        double sim_time_delta;
        double discount;

        State init_state;
        vector<State> sampled_controls;
        kdtree* controls_tree; 
        vector<State> sampled_observations;

        System(double discount_factor, double process_noise_in);
        ~System();

        // functions
        
        double get_holding_time(State& s, State& control, double gamma, int num_vert)
        {
            State max_control;
            State min_state;
            for(int i=0; i<NUM_DIM; i++)
            {
                max_control.x[i] = max_controls[i];
                min_state.x[i] = min_states[i];
            }

            State absf = max_control - min_state;

            double h = max(gamma * pow( log(num_vert+1.0)/(num_vert+1.0), 1.0/(double)NUM_DIM), 1e-3);
            double num = h*h;
            
            for(int i=0; i<NUM_DIM; i++)
                num = num*sq(max_states[i] - min_states[i]);

            double sqnum = sqrt(num);
            double den = 0;
            for(int i=0; i< NUM_DIM; i++)
                den += process_noise[i];
            
            den += (sqnum*absf.norm());
            
            double tmp = num/den;
            if(tmp < 1e-3)
            {
                cout<<"holding time too less " << num << " " << den << endl;
                getchar();
            }
            return num/(den);
        }
        
        int get_key(State& s, double *key)
        {
            for(int i =0; i < NUM_DIM; i++)
            {
                key[i] = (s.x[i] - min_states[i])/(max_states[i] - min_states[i]);
                //assert(key[i] <= 1.1);
            }
            return 0;

        };
        bool is_free(State &s)
        {
            return 1;
        };
        
        State sample_init_state()
        {
            State s;
            multivar_normal(init_state.x, init_var, s.x, NUM_DIM);

            return s;
        }

        State sample_state()
        {
            State s;
            double *max_p;
            double *min_p;
            double prob = RANDF;
            if(prob < 0.1)
            {
                max_p = max_goal;
                min_p = min_goal;
            }
            else
            {
                max_p = max_states;
                min_p = min_states;
            }
            while(1)
            {
                for(int i=0; i< NUM_DIM; i++)
                {
                    s.x[i] = min_p[i] + RANDF*( max_p[i] - min_p[i]);
                }

                if( is_free(s) )
                    break;
            }
            return s;

        }
        
        State sample_control()
        {
            State c;
            for(int i=0; i< NUM_DIM; i++)
            {
                c.x[i] = min_controls[i] + RANDF*( max_controls[i] - min_controls[i]);
            }

            return c;
        }
        State sample_observation()
        {
            State s;
            double *max_p = max_states;
            double *min_p = min_states;
            while(1)
            {
                for(int i=0; i< NUM_DIM; i++)
                {
                    s.x[i] = min_p[i] + RANDF*( max_p[i] - min_p[i]);
                }

                if( is_free(s) )
                    break;
            }

            State obs_s = observation(s, false);
            return obs_s;
        }

        State sample_goal()
        {
            State g;
            for(int i=0; i< NUM_DIM; i++)
            {
                g.x[i] = min_goal[i] + RANDF*( max_goal[i] - min_goal[i]);
            }
            return g;
        }
        bool is_inside_goal(State& s)
        {
            for(int i=0; i< NUM_DIM; i++)
            {
                if( (s.x[i] > max_goal[i]) || (s.x[i] < min_goal[i]) )
                    return false;
            }
            return true;
        }

        void get_obs_variance(State& s, double* var);
        void get_variance(State& s, double duration, double* var);
        State get_fdt(State& s, State& control, double duration);
        State get_controller(State& s);
        State integrate(State& s, State& control, double duration, bool is_clean);
        State integrate_alpha(double alpha, State& s, State& control, double duration, bool is_clean);
        State observation(State& s, bool is_clean);
        int sample_control_observations(int num_vert);
        
        int get_lgq_path(double dT, vector<State>& lgq_path, vector<State>& lqg_covar, \
        vector<State>& lqg_control, double& total_cost);
};



#endif
