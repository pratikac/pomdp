#ifndef __mdp_h__
#define __mdp_h__

#include "utils/common.h"
#include "systems/singleint.h"

class Edge;
class Vertex;
class Graph;

class Vertex
{
    public:

        State s;
        
        double holding_time;
        
        vector<State *> state_obs;
        vector<State *> controls;

        list<Edge *> edges_in;
        list<Edge *> edges_out;
        
        // stores the position of the end-marker for edges corresponding to particular control
        // used while normalizing
        vector<int>controls_iter;

        Vertex(State& st);
        ~Vertex(); 
};

class Edge{

    public:
        Vertex *from;
        Vertex *to;
        
        // constant control for transition (from -- to)
        // applied for time = transition_time
        State *control;

        list<Edge*>::iterator from_iter;
        list<Edge*>::iterator to_iter;
        list<Edge*>::iterator elist_iter;

        double transition_prob;
        double transition_time;
        
        Edge(Vertex* f, Vertex* t, double prob, double trans_time);

        Edge reverse(){
            return Edge(this->to, this->from, this->transition_prob, this->transition_time);
        }
        ~Edge()
        {
            //cout<<"called destructor"<<endl;

        };
};

class Graph{

    public:
        
        bot_lcmgl_t *lcmgl;
        
        double delta;
        double min_holding_time;
        bool seeding_finished;

        double gamma, gamma_t;
        struct kdtree *state_tree;
        struct kdtree *obs_tree;
       
        System* system;

        Graph(System& sys, bot_lcmgl_t *in_lcmgl);
        ~Graph();
        
        vector<Vertex *> vlist;
        list<Edge *> elist;
        
        unsigned int num_vert;
                
        // graph sanity check
        list< list<State> > monte_carlo_trajectories;
        list<double> monte_carlo_probabilities;
        list< list<double> > monte_carlo_times;

        // graph functions
        unsigned int get_num_vert(){return num_vert; };

        int vertex_delete_edges(Vertex* v);
        void remove_edge(Edge *e);
        
        int insert_into_state_tree(Vertex *v)
        {
            double key[NUM_DIM];
            system->get_key(v->s, key);
            kd_insert(state_tree, key, v);
            
            return 0;
        };

        int insert_into_obs_tree(State *s)
        {
            double key[NUM_DIM];
            system->get_key(*s, key);
            kd_insert(obs_tree, key, s);
            
            return 0;
        };

        Vertex* nearest_vertex(State s);
        void normalize_edges(Vertex *from);
 
        double dist(State s1, State s2)
        {
            double t = 0;
            for(int i=0; i<NUM_DIM; i++)
                t = t + (s1.x[i] - s2.x[i])*(s1.x[i] - s2.x[i]);

            return sqrt(t);
        };       
        
        void print_rrg();
        void plot_graph();
        
        int simulate_trajectory();
        void plot_monte_carlo_trajectories(); 
        void plot_monte_carlo_density(char* filename);

        // algorithm functions
        
        int add_sample();
        bool is_edge_free( Edge *etmp);
        
        int reconnect_edges_neighbors(Vertex* v);
        int connect_edges(Vertex *v);
        int connect_edges_approx(Vertex *v);

        void put_init_samples(int howmany);
       
        bool is_everything_normalized();

        int simulate_trajectory(double duration);
};

class MDP{

    public:
         
        MDP(Graph& in_graph, bot_lcmgl_t *in_lcmgl);
        ~MDP();

        bot_lcmgl_t *lcmgl;
        Graph *graph;

        double max_obs_time;
        
        vector<State> control;
        list<State> truth;
        int obs_curr_index;
        vector<double> obs_times;
        vector<State> obs;
        
        list<State> best_path;

        // functions
        int simulate_trajectory()
        {
            graph->simulate_trajectory(max_obs_time);
            return 0;
        }
        void propagate_system();
        
        void draw_lcm_grid();
        void plot_trajectory();
};



#endif
