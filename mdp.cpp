#include "mdp.h"

Vertex::Vertex(State& st)
{
    s = st;
}

Vertex::~Vertex()
{
    for(vector<State *>::iterator i= state_obs.begin(); i != state_obs.end(); i++)
        delete (*i);

    //controls vector is built from pointers, which are cleared by the class system
    controls.clear();

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
    max_obs_time = 1.0;
};

MDP::~MDP()
{
    control.clear();
    truth.clear();
    obs.clear();
    obs_times.clear();
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

void MDP::write_pomdp_file()
{
    System *sys = graph->system;

    ofstream sindex("sarsop/state_index.dat");

    for(int i=0; i< graph->num_vert; i++)
    {
        sindex<<i<<"\t";
        graph->vlist[i]->s.print(sindex);
    }
    sindex.close();

    ofstream pout("sarsop/problem.pomdp");
    pout <<"#This is an auto-generated pomdp file from the MDP\n" << endl;
    pout <<"discount: 0.99" << endl;
    pout <<"values: reward" << endl;
    pout <<"states: "<< graph->num_vert << endl;
    pout <<"actions: "<< graph->num_sampled_controls << endl;
    pout <<"observations: "<< graph->num_vert << endl;

    pout << endl;

    pout <<"start: ";
    vector<double> tmp;
    double totprob = 0;
#if 1
    for(int i=0; i< graph->num_vert; i++)
    {
        double prior = normal_val(sys->init_state.x, sys->init_var, graph->vlist[i]->s.x, NUM_DIM);
        totprob += prior;
        tmp.push_back(prior);
    }
    
    for(int i=0; i< graph->num_vert; i++)
    {
        pout<< tmp[i]/totprob<<" ";
    }
#else
    pout <<"uniform" << endl;
#endif
    pout<<endl << endl;
    
    pout<<"#Transition probabilities"<<endl;
    for(int i=0; i<graph->num_vert; i++)
    {
        Vertex *vtmp = graph->vlist[i];
        if(!sys->is_inside_goal(vtmp->s))
        {
            for(list<Edge*>::iterator j= vtmp->edges_out.begin(); j != vtmp->edges_out.end(); j++)
            {
                Edge *etmp = (*j);
                pout <<"T: "<< etmp->control_index <<" : "<< etmp->from->index_in_vlist << " : "\
                    << etmp->to->index_in_vlist <<" " << etmp->transition_prob << endl;
            }
        }
        else
        {
            pout <<"T: *" <<" : "<< vtmp->index_in_vlist << " : "\
                << vtmp->index_in_vlist <<" " << 1 << endl;
        }
    }
    pout<< endl << endl;

    pout <<"#Observation probabilities" << endl;
    for(int i=0; i< graph->num_vert; i++)
    {
        Vertex *v1 = graph->vlist[i];
        State obs_v1 = sys->observation(v1->s, true);
        
        totprob = 0;
        tmp.clear();
        for(int j=0; j < graph->num_vert; j++)
        {
            Vertex *v2 = graph->vlist[j];
            State obs_v2 = sys->observation(v2->s, true);
            double local_obs_variance[NUM_DIM_OBS] = {0};
            sys->get_obs_variance(v1->s, local_obs_variance);
            double ptmp = normal_val( obs_v1.x, local_obs_variance, obs_v2.x, NUM_DIM_OBS);
            totprob += ptmp;

            tmp.push_back(ptmp);
        }
        pout <<"O: * : " << i <<" ";
        for(int j=0; j < graph->num_vert; j++)
        {
#if 0
            if( j != i)
                pout << 0 << " ";
            else
                pout << 1 << " ";
#else
            pout << tmp[j]/totprob <<" ";
#endif
        }
        pout<<endl;
    }
    pout << endl;
    
    pout <<"#Rewards" << endl;

    for(int i =0; i< graph->num_sampled_controls; i++)
    {
        for(int j=0; j< graph->num_vert; j++)
        {
            Vertex *v1 = graph->vlist[j];
#if 1
            if(! sys->is_inside_goal(v1->s))
            {
                pout <<"R: " << i <<" : * : "<< j << " : * " <<   -0*(v1->s).norm2()\
                    -0*(sys->sampled_controls[i]).norm2()  << endl;
            }
            else
            {
                pout <<"R: " << i <<" : * : "<< j << " : * " << 100  << endl;
            }
#else
            pout <<"R: " << i <<" : * : "<< j << " : * " << -(v1->s).norm2() + \
                    -(sys->sampled_controls[i]).norm2()  << endl;
#endif
        }

    }
    pout << endl;

    pout.close();
}


Graph::Graph(System& sys, bot_lcmgl_t *in_lcmgl) 
{
    lcmgl = in_lcmgl;

    system = &sys;
   
    num_sampled_controls = system->sampled_controls.size();

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
    
    factor = 1;
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
    //cout<<"delete_edges before: " << v->edges_out.size() << endl;
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
    //cout<<"delete_edges after: " << v->edges_out.size() << endl;
    
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
    
    
    //cout<<"rrg size: "<< vlist.size() << endl;
    for(vector<Vertex*>::iterator i = vlist.begin(); i != vlist.end(); i++)
    {
        Vertex *tstart = (*i);
#if 1
        for(list<Edge*>::iterator eo = tstart->edges_out.begin(); eo != tstart->edges_out.end(); eo++)
        {
            Vertex *tend = (*eo)->to;

            //draw the edge
            
            bot_lcmgl_begin(lcmgl, GL_LINES);
            double toput1[3] ={0};
            for(int i =0; i< NUM_DIM; i++)
                toput1[i] = tstart->s.x[i];
            bot_lcmgl_vertex3f(lcmgl, toput1[0], toput1[1], toput1[2]);
            
            for(int i =0; i< NUM_DIM; i++)
                toput1[i] = tend->s.x[i];
            bot_lcmgl_vertex3f(lcmgl, toput1[0], toput1[1], toput1[2]);
            bot_lcmgl_end(lcmgl);

        }
#endif
        bot_lcmgl_begin(lcmgl, GL_POINTS);
        double toput[3] ={0};
        for(int i =0; i< NUM_DIM; i++)
            toput[i] = tstart->s.x[i];

        bot_lcmgl_vertex3f(lcmgl, toput[0], toput[1], toput[2]);
        bot_lcmgl_end(lcmgl);
    }
#endif
}

void MDP::plot_trajectory()
{
#if 1
    double curr_time =0;
    bot_lcmgl_color4f(lcmgl, 1, 0, 0, 1);
    bot_lcmgl_point_size(lcmgl, 4.0);
    
    bot_lcmgl_begin(lcmgl, GL_LINES);
    for(vector<State>::iterator i= truth.begin(); i != truth.end(); i++)
    {
        State& curr = *i;
        vector<State>::iterator j = i;
        j++;

        double toput1[3] ={0};
        for(int i =0; i< NUM_DIM; i++)
            toput1[i] = curr.x[i];

        bot_lcmgl_vertex3f(lcmgl, toput1[0], toput1[1], toput1[2]);
        
        if( (j) != truth.end() )
        {
            State& next = *(j);
        
            double toput2[3] ={0};
            for(int i =0; i< NUM_DIM; i++)
                toput2[i] = next.x[i];

            bot_lcmgl_vertex3f(lcmgl, toput2[0], toput2[1], toput2[2]);
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
        
        double toput1[3] ={0};
        for(int i =0; i< NUM_DIM; i++)
            toput1[i] = curr.x[i];

        bot_lcmgl_vertex3f(lcmgl, toput1[0], toput1[1], toput1[2]);
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

int Graph::make_holding_time_constant_all()
{
    State holding_state, holding_control;
    for(int i=0; i<NUM_DIM; i++)
    {
        holding_state.x[i] = system->min_states[i];
        holding_control.x[i] = system->max_controls[i];
    }

    constant_holding_time = 1000;
    for(int i=0; i< num_vert; i++)
    {
        Vertex* vtmp = vlist[i];
        for(unsigned int j=0; j < vtmp->holding_times.size(); j++)
        {
            if(vtmp->holding_times[j] < constant_holding_time)
                constant_holding_time = vtmp->holding_times[j];
        }
    }
    constant_holding_time = constant_holding_time/10;

    cout<<"delta: " << constant_holding_time << endl;
    for(int i=0; i< num_vert; i++)
    {
        make_holding_time_constant(vlist[i]);
    }
    return 0;
}
int Graph::make_holding_time_constant(Vertex* from)
{
    list<Edge *>::iterator last_control_iter = from->edges_out.begin();
    int controls_iter_iter = 0;
    int edges_num = 0;

    for(list<Edge *>::iterator i = from->edges_out.begin(); i != from->edges_out.end(); i ++)
    {
        edges_num++;
        
        list<Edge *>::iterator i_plus_one = i;
        i_plus_one++;

        if( edges_num == from->controls_iter[controls_iter_iter] )
        {
            // cout<<"controls_iter_iter: " << from->controls_iter[controls_iter_iter] << endl;
            // normalize between last_edge_count and edge_count
            
            // add new edge
            double pself = 1 - constant_holding_time/from->holding_times[controls_iter_iter];
            if( (pself > 1) || (pself < 0) )
            {
                cout<<"pself greater: " << pself<<" constant holding_time: " << constant_holding_time << endl;
            }
            from->holding_times[controls_iter_iter] = constant_holding_time;
            
            Edge* eself = new Edge(from, from, pself, constant_holding_time);
            
            eself->control = &(system->sampled_controls[controls_iter_iter]);
            eself->control_index = controls_iter_iter;

            elist.push_back(eself);
            from->edges_out.push_back(eself);
            from->edges_in.push_back(eself);

            eself->elist_iter = elist.end();            eself->elist_iter--;
            eself->from_iter = from->edges_out.end();   eself->from_iter--;
            eself->to_iter = from->edges_in.end();      eself->to_iter--;

            for(list<Edge *>::iterator j = last_control_iter;\
                    j != i_plus_one; j++)
            {
                Edge *etmp = (*j);
                etmp->transition_prob = etmp->transition_prob *(1 - pself);
                etmp->transition_time = constant_holding_time;
                //cout<<"wrote edge prob: "<< etmp->transition_prob << endl;
            }
                        
            last_control_iter = i_plus_one;      // edges belonging to new control start from here
            controls_iter_iter++;
        }
    }
    return 0;
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
        totprob += (*i)->transition_prob;
        
        list<Edge *>::iterator i_plus_one = i;
        i_plus_one++;

        if( edges_num == from->controls_iter[controls_iter_iter] )
        {
            // cout<<"controls_iter_iter: " << from->controls_iter[controls_iter_iter] << endl;
            // normalize between last_edge_count and edge_count
            
            if(totprob > 1.0/DBL_MAX)
            {
                for(list<Edge *>::iterator j = last_control_iter;\
                        j != i_plus_one; j++)
                {
                    Edge *etmp = (*j);
                    etmp->transition_prob = etmp->transition_prob / totprob;
                    //cout<<"wrote edge prob: "<< etmp->transition_prob << endl;
                }
            }
            else
            {
                for(list<Edge *>::iterator j = last_control_iter;\
                        j != i; j++)
                {
                    Edge *etmp = (*j);
                    etmp->transition_prob = 1;
                    cout<<"setting prob to 1, maybe only one edge" << endl;
                    //cout<<"wrote edge prob: "<< etmp->transition_prob << endl;
                }
                //cout<<"totprob is: "<< totprob << " [DITCH]" << endl;
            }
            
            last_control_iter = i_plus_one;      // edges belonging to new control start from here
            totprob = 0;
            controls_iter_iter++;
        }
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

int Graph::add_sample(bool is_init)
{
    State stmp;
    if (is_init == true)
        stmp = system->sample_init_state();
    else
        stmp = system->sample_state();

    Vertex *v = new Vertex(stmp);
    
    if(num_vert == 0)
    {
        v->index_in_vlist = 0;

        vlist.push_back(v);
        num_vert++;
        insert_into_state_tree(v);
    
#if 0
        for(int i=0; i<50; i++)
        {
            State *sobs = new State();
            *sobs = system->observation(v->s, false);
            v->state_obs.push_back(sobs);
            insert_into_obs_tree(sobs);
        }
#endif
    }
    else
    {
        if( connect_edges_approx(v) == 0 )
        {
            v->index_in_vlist = num_vert;

            vlist.push_back(v);
            num_vert++;
            insert_into_state_tree(v);
#if 0
            for(int i=0; i<50; i++)
            {
                State *sobs = new State();
                *sobs = system->observation(v->s, false);
                v->state_obs.push_back(sobs);
                insert_into_obs_tree(sobs);
            }
#endif
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
            
            //cout<<"reconnecting vertex: " << v1->index_in_vlist << endl;
            vertex_delete_edges(v1);
            connect_edges_approx(v1);
        }

        kd_res_next(res);
    }
    kd_res_free(res);
    
#endif

#if 0
    for(list<Edge*>::iterator i = v->edges_out.begin(); i != v->edges_out.end(); i++)
    {
        Vertex* v1 = (*i)->to;
        cout<<"reconnecting vertex: " << v1->index_in_vlist << endl;
        vertex_delete_edges(v1);
        connect_edges_approx(v1);
    }
#endif

    return 0;
}

int Graph::connect_edges_approx(Vertex* v)
{
    // don't draw outgoing edges to goal
    if( system->is_inside_goal(v->s) )
        return 0;

    v->controls.clear();
    v->controls_iter.clear();
    v->edges_out.clear();

    double key[NUM_DIM] ={0};
    system->get_key(v->s, key);

    double bowlr = max(gamma * pow( log(num_vert+1.0)/(double)(num_vert+1.0), 1.0/(double)NUM_DIM), 1e-3);
    // cout<<"bowlr: " << bowlr << endl;

    kdres *res;
    res = kd_nearest_range(state_tree, key, bowlr );

    if(kd_res_size(res) == 0)
        return 1;

    //cout<<"got "<<kd_res_size(res)<<" states in bowlr = "<< bowlr << endl;
    //int pr = kd_res_size(res);

    double *sys_var = new double[NUM_DIM];
    
    int num_edges = 0;
    for(int i = 0; i< num_sampled_controls; i++)
    {
        State *curr_control = &(system->sampled_controls[i]);
        v->controls.push_back( curr_control );

        double holding_time = system->get_holding_time(v->s, *curr_control, gamma, num_vert);
        v->holding_times.push_back(holding_time);
        //cout<<"ht: "<< holding_time << endl;

        State stmp = system->get_fdt(v->s, *curr_control, holding_time);
        system->get_variance(v->s, holding_time, sys_var);

        double pos[NUM_DIM] = {0};
        while( !kd_res_end(res) )
        {
            Vertex* v1 = (Vertex* ) kd_res_item(res, pos);

            if(v1 != v)
            {
                double prob_tmp = normal_val(stmp.x, sys_var, v1->s.x, NUM_DIM);
                if(prob_tmp > 0)
                {
                    Edge *e1 = new Edge(v, v1, prob_tmp, holding_time);
                    e1->control = curr_control;
                    e1->control_index = i;

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
            cout<<"\t "<< (*j)->from->index_in_vlist <<"-"<< (*j)->to->index_in_vlist << \
                " control: " << (*j)->control_index << " " << (*j)->transition_prob << endl;
        }

        counte = 0;
        double totprob = 0;
        cout<<"eo: " << endl;
        for(list<Edge *>::iterator j = v->edges_out.begin(); j != v->edges_out.end(); j++)
        {
            cout<<"\t "<< (*j)->from->index_in_vlist <<"-"<< (*j)->to->index_in_vlist << \
                " contol: " << (*j)->control_index << " " << (*j)->transition_prob << endl;
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
        add_sample(false);
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
#if 1
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
    bot_lcmgl_color4f(lcmgl, 1, 0, 0.5, 0.5);

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
            double toput[3] ={0};
            for(int i =0; i< NUM_DIM; i++)
                toput[i] = curr_state.x[i];

            bot_lcmgl_vertex3f(lcmgl, toput[0], toput[1], toput[2]);
        }
        prob_iter++;
        count++;
    }
    bot_lcmgl_end(lcmgl);
}

