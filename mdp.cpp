#include "mdp.h"

Vertex::Vertex(State& st)
{
    s = st;

    holding_time = -1;

    edges_in.clear();
    edges_out.clear();
}

Vertex::~Vertex()
{
    for(vector<State *>::iterator i= state_obs.begin(); i != state_obs.end(); i++)
        delete (*i);

    for(vector<State *>::iterator i= controls.begin(); i != controls.end(); i++)
        delete (*i);

    state_obs.clear();
    controls.clear();
    controls_iter.clear();

    edges_in.clear();
    edges_out.clear();
};

Edge::Edge(Vertex *f, Vertex *t, double prob, double trans_time){
    from = f;
    to = t;
    transition_prob = prob;
    transition_time = trans_time;
}

MDP::MDP(Graph& in_graph, bot_lcmgl_t *in_lcmgl)
{
    lcmgl = in_lcmgl;
    graph = &in_graph;
    obs_curr_index = 0;
    max_obs_time = 10.0;
};

MDP::~MDP()
{
    control.clear();
    truth.clear();
    obs.clear();
    obs_times.clear();
}

void MDP::propagate_system()
{
    control.clear();
    truth.clear();
    obs.clear();
    obs_times.clear();

    System *sys = graph->system;

    double curr_time = 0;
    double max_time = max_obs_time;

    State zero_control;

    truth.push_back( sys->init_state);
    control.push_back(zero_control);

    while(curr_time < max_time)
    {
        State snext = sys->integrate( truth.back(), sys->sim_time_delta, false);
        truth.push_back( snext);
        control.push_back(zero_control);
        
        State next_obs = sys->observation( snext, false);
        obs.push_back(next_obs);
        obs_times.push_back(curr_time);
        curr_time += sys->sim_time_delta;
    }
}

void MDP::draw_lcm_grid()
{
    bot_lcmgl_color4f(lcmgl, 0, 0, 0, 0.5);
    
    bot_lcmgl_begin(lcmgl, GL_LINES);
    for(double x = -10; x < 10; x += 0.01)
    {
        bot_lcmgl_vertex3f(lcmgl, x, 0, 0);
        bot_lcmgl_vertex3f(lcmgl, x+0.01, 0, 0);
    }
    bot_lcmgl_end(lcmgl);

    bot_lcmgl_begin(lcmgl, GL_LINES);
    for(double y = -10; y < 10; y += 0.01)
    {
        bot_lcmgl_vertex3f(lcmgl, 0, y, 0);
        bot_lcmgl_vertex3f(lcmgl, 0, y+0.01, 0);
    }
    bot_lcmgl_end(lcmgl);
#if 0
    bot_lcmgl_begin(lcmgl, GL_LINES);
    for(double z = -10; z < 10; z += 0.01)
    {
        bot_lcmgl_vertex3f(lcmgl, 0, 0, z);
        bot_lcmgl_vertex3f(lcmgl, 0, 0, z+0.01);
    }
    bot_lcmgl_end(lcmgl);
#endif
};

Graph::Graph(System& sys, bot_lcmgl_t *in_lcmgl) 
{
    lcmgl = in_lcmgl;

    system = &sys;

    vlist.clear();
    num_vert = 0;

    state_tree = kd_create(NUM_DIM);
    obs_tree = kd_create(NUM_DIM);

    double factor = 1;
    if(NUM_DIM == 2)
        factor = M_PI;
    else if(NUM_DIM == 3)
        factor = 4/3*M_PI;
    else if(NUM_DIM == 4)
        factor = 0.5*M_PI*M_PI;

    gamma = 2.1*pow( (1+1/(double)NUM_DIM), 1/(double)NUM_DIM) *pow(factor, -1/(double)NUM_DIM);
};

Graph::~Graph()
{
    for(vector<Vertex*>::reverse_iterator i = vlist.rbegin(); i != vlist.rend(); i++)
    {
        Vertex *vtmp = *i;
        delete vtmp;
        num_vert--;
    }
    for(list<Edge*>::reverse_iterator i = elist.rbegin(); i != elist.rend(); i++)
    {
        delete *i;
    }
    
    vlist.clear();

    kd_free(state_tree);
    kd_free(obs_tree);
}

int Graph::vertex_delete_edges(Vertex* v)
{
    for(list<Edge *>::reverse_iterator i = v->edges_out.rbegin(); i != v->edges_out.rend(); i++)
    {
        //debug
        
        Edge* etmp = (*i);
        //cout<<"elist: " << distance(elist.begin(), etmp->elist_iter) << endl;
        //cout<<"to_list: " << distance(etmp->to->edges_in.begin(), etmp->to_iter) << endl;
        //cout<<"from_list: " << distance(etmp->from->edges_out.begin(), etmp->from_iter) << endl;
        
        elist.erase(etmp->elist_iter);
        etmp->to->edges_in.erase(etmp->to_iter);
        delete etmp;
    }
    v->edges_out.clear();
    return 0;
}

void Graph::remove_edge(Edge *e)
{
    elist.erase(e->elist_iter);

    //cout<<"--- inside remove_edge" << endl;
    // remove from, from + to lists and then call destructor
    e->from->edges_out.erase(e->from_iter);

    e->to->edges_in.erase(e->to_iter);
    //cout<<"removed to, delete" << endl;

    delete e;
}

void Graph::plot_graph()
{
#if 1
    bot_lcmgl_color4f(lcmgl, 0, 1, 0, 0.5);
    bot_lcmgl_point_size(lcmgl, 4.0);
    
    bot_lcmgl_begin(lcmgl, GL_POINTS);
    
    //cout<<"rrg size: "<< vlist.size() << endl;
    for(vector<Vertex*>::iterator i = vlist.begin(); i != vlist.end(); i++)
    {
        Vertex *tstart = (*i);
#if 0
        for(list<Edge*>::iterator eo = tstart->edges_out.begin(); eo != tstart->edges_out.end(); eo++)
        {
            Vertex *tend = (*eo)->to;
            Edge *etmp = (*eo);

            //draw the edge
            rrgout<<tstart->s.x[0]<<"\t"<<tstart->s.x[1]<<"\t"<<tend->s.x[0]<<"\t"<<tend->s.x[1]<<"\t"<<etmp->transition_prob<<"\t"<<etmp->transition_time<<endl;
        }
#endif
        bot_lcmgl_vertex3f(lcmgl, tstart->s.x[0], tstart->s.x[1], 0);
    }
    bot_lcmgl_end(lcmgl);
#endif
}

void MDP::plot_trajectory()
{
#if 1
    double curr_time =0;
    bot_lcmgl_color4f(lcmgl, 1, 0, 0, 1);
    bot_lcmgl_point_size(lcmgl, 4.0);
    
    bot_lcmgl_begin(lcmgl, GL_LINES);
    for(list<State>::iterator i= truth.begin(); i != truth.end(); i++)
    {
        State& curr = *i;
        list<State>::iterator j = i;
        j++;

        bot_lcmgl_vertex3f(lcmgl, curr.x[0], curr.x[1], 0);
        
        if( (j) != truth.end() )
        {
            State& next = *(j);
            bot_lcmgl_vertex3f(lcmgl, next.x[0], next.x[1], 0);
        }
        curr_time += graph->system->sim_time_delta;
    }
    bot_lcmgl_end(lcmgl);

#if 0
    count = 0;
    bot_lcmgl_color4f(lcmgl, 0, 0, 1, 0.3);
    bot_lcmgl_point_size(lcmgl, 4.0);
    
    bot_lcmgl_begin(lcmgl, GL_POINTS);
    for(vector<State>::iterator i= obs.begin(); i != obs.end(); i++)
    {
        State& curr = *i;
        bot_lcmgl_vertex3f(lcmgl, curr.x[0], curr.x[1], 0);
        count++;
    }
    bot_lcmgl_end(lcmgl);
#endif
#endif
}


Vertex* Graph::nearest_vertex(State s)
{
    assert(num_vert > 0);
    double key[NUM_DIM];
    system->get_key(s, key);

    kdres *res;
    res = kd_nearest(state_tree, key);
    if(kd_res_end(res))
    {
        cout<<"Error: no nearest"<<endl;
        exit(1);
    }
    Vertex *v = (Vertex*)kd_res_item_data(res);
    kd_res_free(res);

    return v;
}

void Graph::normalize_edges(Vertex *from)
{
    list<Edge *>::iterator last_control_iter = from->edges_out.begin();
    int controls_iter_iter = 0;
    double totprob = 0;
    int edges_num = 0;

    for(list<Edge *>::iterator i = from->edges_out.begin(); i != from->edges_out.end(); i ++)
    {
        edges_num++;

        if( edges_num == from->controls_iter[controls_iter_iter] )
        {
            // normalize between last_edge_count and edge_count
            
            if(totprob > 1.0/DBL_MAX)
            {
                for(list<Edge *>::iterator j = last_control_iter;\
                        j != i; j++)
                {
                    Edge *etmp = (*j);
                    etmp->transition_prob = etmp->transition_prob / totprob;
                    //cout<<"wrote edge prob: "<< etmp->transition_prob << endl;
                }
            }
            else
            {
                //cout<<"totprob is: "<< totprob << " [DITCH]" << endl;
            }
            
            list<Edge *>::iterator tmp = i;
            tmp++;
            last_control_iter = tmp;      // edges belonging to new control start from here
            totprob = 0;
            controls_iter_iter++;
        }
        totprob += (*i)->transition_prob;
    }

#if 0
    if(etmp->transition_prob != etmp->transition_prob)
    {
        cout<<"found a nan: "<< totprob << " nedges: "<< nedges << endl;
        getchar();
    }
#endif

    /*
       cout<<"getchar(): ";
       getchar();
       cout<<endl;
       */
}

bool Graph::is_edge_free( Edge *etmp)
{
    return true;
#if 0
    int num_discrete = 10;
    State& init = etmp->from->s;
    State& end = etmp->to->s;

    for(int i=0; i< num_discrete+1; i++)
    {
        State stmp;
        for(int j=0; j< NUM_DIM; j++)
            stmp.x[j] = init.x[j] + (end.x[j] - init.x[j])/num_discrete*i;

        if( system->is_free(stmp) == 0)
            return false;
    }
    return true;
#endif
}

int Graph::add_sample()
{
    State stmp = system->sample_state();
    Vertex *v = new Vertex(stmp);
    
    if(num_vert == 0)
    {
        vlist.push_back(v);
        num_vert++;
        insert_into_state_tree(v);

        for(int i=0; i<50; i++)
        {
            State *sobs = new State();
            *sobs = system->observation(v->s, false);
            v->state_obs.push_back(sobs);
            insert_into_obs_tree(sobs);
        }
    }
    else 
    {
        if( connect_edges_approx(v) == 0 )
        {
            vlist.push_back(v);
            num_vert++;
            insert_into_state_tree(v);

            for(int i=0; i<50; i++)
            {
                State *sobs = new State();
                *sobs = system->observation(v->s, false);
                v->state_obs.push_back(sobs);
                insert_into_obs_tree(sobs);
            }

            reconnect_edges_neighbors(v);
        }
        else
        {
            delete v;
            return 1;
        }
    }
    return 0;
}

int Graph::reconnect_edges_neighbors(Vertex* v)
{
#if 1
    
    double key[NUM_DIM] ={0};
    system->get_key(v->s, key);

    double bowlr = gamma/2 * pow( log(num_vert)/(num_vert), 1.0/(double)NUM_DIM);

    kdres *res;
    res = kd_nearest_range(state_tree, key, bowlr );
    //int pr = kd_res_size(res);
    //cout<<"reconnecting "<<kd_res_size(res)<<" nodes, radius: " << bowlr << endl;

    double pos[NUM_DIM] = {0};
    while( !kd_res_end(res) )
    {
        Vertex* v1 = (Vertex* ) kd_res_item(res, pos);

        if(v1 != v)
        {
            // remove old edges
            
            vertex_delete_edges(v1);
            connect_edges_approx(v1);
        }

        kd_res_next(res);
    }
    kd_res_free(res);
    
#endif

#if 0
    Vertex* v1 = nearest_vertex(v->s);
    vertex_delete_edges(v1);
    connect_edges_approx(v1);
#endif

    return 0;
}

int Graph::connect_edges_approx(Vertex* v)
{
    double key[NUM_DIM] ={0};
    system->get_key(v->s, key);

    double bowlr = gamma * pow( log(num_vert+1.0)/(double)(num_vert+1.0), 1.0/(double)NUM_DIM);
    //cout<<"bowlr: " << bowlr << endl;

    double holding_time = system->get_holding_time(v->s, gamma, num_vert);
    //cout<< holding_time << endl;
    v->holding_time = holding_time;

    kdres *res;
    res = kd_nearest_range(state_tree, key, bowlr );

    if(kd_res_size(res) == 0)
        return 1;

    //cout<<"got "<<kd_res_size(res)<<" states in bowlr= "<< bowlr << endl;
    //int pr = kd_res_size(res);

    double *sys_var = new double[NUM_DIM];
    system->get_variance(v->s, holding_time, sys_var);
    State stmp = system->get_fdt(v->s, holding_time);
    
    int num_edges = 0;
    for(int i = 0; i<3; i++)
    {
        State *control_tmp = new State();
        *control_tmp = system->sample_control();
        v->controls.push_back(control_tmp);

        for(int dim=0; dim<NUM_DIM; dim++)
            control_tmp->x[dim] = control_tmp->x[dim] + stmp[dim];

        double sum_prob = 0;
        double pos[NUM_DIM] = {0};
        while( !kd_res_end(res) )
        {
            Vertex* v1 = (Vertex* ) kd_res_item(res, pos);

            if(v1 != v)
            {
                double prob_tmp = normal_val(control_tmp->x, sys_var, v1->s.x, NUM_DIM);
                if(prob_tmp > 0)
                {
                    Edge *e1 = new Edge(v, v1, prob_tmp, holding_time);
                    e1->control = control_tmp;

                    sum_prob += prob_tmp;

                    num_edges++;
                    elist.push_back(e1);
                    v->edges_out.push_back(e1);
                    v1->edges_in.push_back(e1);

                    e1->elist_iter = elist.end();       e1->elist_iter--;
                    e1->from_iter = v->edges_out.end(); e1->from_iter--;
                    e1->to_iter = v1->edges_in.end();   e1->to_iter--;
                }
            }
            kd_res_next(res);
        }
        
        kd_res_rewind(res);
        v->controls_iter.push_back(num_edges);      // this is the number of edges
    }
    kd_res_free(res);

    normalize_edges(v);

    delete[] sys_var;

    return 0;
}

void Graph::print_rrg()
{
#if 1
    int counte = 0;
    int countv = 0;
    for(vector<Vertex*>::iterator i = vlist.begin(); i != vlist.end(); i++)
    {
        Vertex *v = *i;
        cout<<"node: " << countv++ << " state: " << v->s.x[0] << " " << v->s.x[1] << endl;

        counte = 0;
        cout << "ei: " << endl;
        for(list<Edge *>::iterator j = v->edges_in.begin(); j != v->edges_in.end(); j++)
        {
            cout<<"\t "<< counte++ << " " << (*j)->transition_prob << endl;
        }

        counte = 0;
        double totprob = 0;
        cout<<"eo: " << endl;
        for(list<Edge *>::iterator j = v->edges_out.begin(); j != v->edges_out.end(); j++)
        {
            cout<<"\t "<< counte++ << " " << (*j)->transition_prob << endl;
            totprob += (*j)->transition_prob;
        }
        cout<<"totprob: "<< totprob << endl;
    }
#endif
}



void Graph::put_init_samples(int howmany)
{
    for(int i=0; i < howmany; i++)
    {
        add_sample();
    }
}

bool Graph::is_everything_normalized()
{
    for(vector<Vertex*>::iterator i = vlist.begin(); i != vlist.end(); i++)
    {
        Vertex* v = *i;
        if(v->edges_out.size() > 0)
        {
            double totprob = 0;
            for(list<Edge *>::iterator j = v->edges_out.begin(); j != v->edges_out.end(); j++)
            {
                totprob += (*j)->transition_prob;
            }

            if( fabs(totprob - 1.0) > 0.1)
            {
                cout<<"offending vertex prob: "<< totprob << " nedges: "<< v->edges_out.size() << endl;
                return false;        
            }
        }
    }
    return true;
}

// returns 
int Graph::simulate_trajectory(double duration)
{
#if 0
    list<State> seq;
    list<double> seq_times;

    State stmp = system->init_state;
    multivar_normal(system->init_state.x, system->init_var, stmp.x, NUM_DIM);

    bool can_go_forward = false;

    Vertex *vcurr = nearest_vertex(stmp);

    if(vcurr->edges_out.size() > 0)
        can_go_forward = true;
    else
        cout<<"vcurr has zero edges" << endl;

    seq.push_back(vcurr->s);
    seq_times.push_back(0);

    // keep in multiplicative form for simulation
    double traj_prob = 1;
    double curr_time = 0;
    double max_time = duration;

    while(can_go_forward)
    {
        double rand_tmp = RANDF;
        double runner = 0;
        Edge *which_edge = NULL;
        for(list<Edge*>::iterator eo = vcurr->edges_out.begin(); eo != vcurr->edges_out.end(); eo++)
        {
            runner += (*eo)->transition_prob;
            if(runner > rand_tmp)
            {
                vcurr = (*eo)->to;
                which_edge = *eo;
                break;
            }
        }

        if(vcurr->edges_out.size() == 0)
        {
            can_go_forward = false;
            //cout<<"break line 684 size: " << seq.size() << endl;
            break;
        }

        if(which_edge != NULL)
        {
            seq.push_back(vcurr->s);
            seq_times.push_back(curr_time);
            traj_prob = traj_prob * which_edge->transition_prob;

            curr_time += which_edge->transition_time;
            if(curr_time > max_time)
            {
                //cout<<"finished one sim" << endl;
                break;
            }
        }
        //cout<<curr_time << " ";
    }

    if(1)
    {
        //cout<<"traj_prob: "<<traj_prob << endl; 
        monte_carlo_times.push_back(seq_times);
        monte_carlo_trajectories.push_back( seq);
        monte_carlo_probabilities.push_back(traj_prob);
        return 0;
    }
#endif
    return 1;
}


void Graph::plot_monte_carlo_trajectories()
{
    ofstream mout("data/monte_carlo.dat");
    int count = 0;

    double totprob = 0;
    for(list<double>::iterator i = monte_carlo_probabilities.begin(); \
            i!= monte_carlo_probabilities.end(); i++)
    {
        totprob += (*i);
    }

    list<double>::iterator prob_iter = monte_carlo_probabilities.begin();
    list< list<double> >::iterator times_iter = monte_carlo_times.begin();

    for(list< list<State> >::iterator i= monte_carlo_trajectories.begin(); \
            i != monte_carlo_trajectories.end(); i++)
    {
        list<State> curr_traj = (*i);
        double curr_prob = (*prob_iter)/totprob;

        list<double> time_seq = (*times_iter);
        list<double>::iterator time_seq_iter = time_seq.begin();
        mout<<count<<"\t"<< curr_prob <<"\t"<<endl;
        for(list<State>::iterator j = curr_traj.begin(); j != curr_traj.end(); j++)
        {
            mout<< (*time_seq_iter) <<"\t";
            State& curr_state = *j;
            for(int k=0; k< NUM_DIM; k++)
            {
                mout<< curr_state.x[k]<<"\t";
            }
            for(int k=NUM_DIM; k< 4; k++)
            {
                mout<< 0 <<"\t";
            }

            mout<<endl;
            time_seq_iter++;
        }

        prob_iter++;
        times_iter++;
        count++;
    }

    mout.close();
} 

void Graph::plot_monte_carlo_density(char* filename)
{
    bot_lcmgl_point_size(lcmgl, 10.0);
    
    bot_lcmgl_begin(lcmgl, GL_POINTS);

    int count = 0;

    double totprob = 0;
    for(list<double>::iterator i = monte_carlo_probabilities.begin(); \
            i!= monte_carlo_probabilities.end(); i++)
    {
        totprob += (*i);
    }
    //cout<<"totprob: "<< totprob << endl;

    list<double>::iterator prob_iter = monte_carlo_probabilities.begin();
    for(list< list<State> >::iterator i= monte_carlo_trajectories.begin(); \
            i != monte_carlo_trajectories.end(); i++)
    {
        list<State> curr_traj = (*i);
        double curr_prob = (*prob_iter);
        curr_prob = curr_prob/totprob; 
        //cout<<"curr_prob: "<< curr_prob << endl;

        State& curr_state = curr_traj.back();
        
        if(curr_prob > 1e-2)
        {
            bot_lcmgl_color4f(lcmgl, 1, 0, 0.5, 0.5);
            bot_lcmgl_vertex3f(lcmgl, curr_state.x[0], curr_state.x[1], 0);
        }
        prob_iter++;
        count++;
    }
    bot_lcmgl_end(lcmgl);
}

