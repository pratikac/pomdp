#include "mdp.h"

int main(int argc, char** argv)
{
    int tot_vert = 20;
    float discount = 0.98;
    double process_noise = 0.01;
    if (argc > 1)
    {
        tot_vert = atoi(argv[1]);
        cout<<"vert: "<< tot_vert << endl;
    }
    if (argc > 2)
    {
        discount = atof(argv[2]);
        cout<<"discount: "<< discount << endl;
    }
    if (argc > 3)
    {
        process_noise = atof(argv[3]);
        cout<<"process_noise: "<< process_noise << endl;
    }
    lcm_t *lcm = bot_lcm_get_global(NULL);
    bot_lcmgl_t *lcmgl = bot_lcmgl_init(lcm, "plotter");
    bot_lcmgl_line_width(lcmgl, 2.0);
   
    srand(0);
    System sys(discount, process_noise);
    sys.sample_control_observations(tot_vert);
    Graph graph(sys, lcmgl, tot_vert);
    MDP mdp(graph, lcmgl);
    mdp.draw_lcm_grid();
   
    tic();
    cout<<"Start sampling" << endl;
    for(int i=0; i < tot_vert; i++)
        mdp.graph->add_sample();
    for(int i=0; i< mdp.graph->num_vert; i++)
    {
        mdp.graph->connect_edges_approx(mdp.graph->vlist[i]);
    }
    mdp.graph->make_holding_time_constant_all();
    toc();
    
    if(sys.name == "lightdark")
        mdp.write_pomdp_file_lightdark();
    else if(sys.name == "singleint")
    {
        mdp.write_pomdp_file_singleint();
        // mdp.run_lqg();
    }
    mdp.plot_trajectory();
    mdp.graph->plot_graph();
    bot_lcmgl_switch_buffer(lcmgl);

    cout<<"Finished" << endl;

    return 0;
}
