#include "mdp.h"

int main(int argc, char** argv)
{
    srand(0);
    int tot_vert = 50;
    if (argc > 1)
        tot_vert = atoi(argv[1]);


    lcm_t *lcm = bot_lcm_get_global(NULL);
    bot_lcmgl_t *lcmgl = bot_lcmgl_init(lcm, "plotter");
    bot_lcmgl_line_width(lcmgl, 2.0);
    
    System sys;
    sys.sample_control_observations(tot_vert);
    Graph graph(sys, lcmgl);
    MDP mdp(graph, lcmgl);
    mdp.draw_lcm_grid();
   
    tic();
    cout<<"Start sampling" << endl;
    for(int i=0; i < tot_vert; i++)
    {
        mdp.graph->add_sample();
        if( (i % 100 == 0) && (i!= 0))
        {
            cout<<i<<" ";
            toc();
        }
        // mdp.graph->print_rrg();
        // cout<<"getchar: "<< endl; getchar();
    }
#if 1
    //mdp.graph->constant_holding_time = 0.001;
    for(int i=0; i< mdp.graph->num_vert; i++)
    {
        mdp.graph->connect_edges_approx(mdp.graph->vlist[i]);
    }
    // mdp.graph->make_holding_time_constant_all();
#else
    mdp.graph->make_holding_time_constant_all();
#endif
    toc();
    
    if(sys.name == "lightdark")
        mdp.write_pomdp_file_lightdark();
    else if(sys.name == "singleint")
    {
        mdp.write_pomdp_file_singleint();
        mdp.run_lqg();
    }
    mdp.plot_trajectory();
    mdp.graph->plot_graph();
    bot_lcmgl_switch_buffer(lcmgl);

    cout<<"Finished" << endl;

    return 0;
}
