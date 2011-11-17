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
    Graph graph(sys, lcmgl);
    MDP mdp(graph, lcmgl);
    mdp.draw_lcm_grid();
   
    tic();
    mdp.graph->add_sample(true);
    mdp.graph->add_sample(true);
    cout<<"Start sampling" << endl;
    for(int i=0; i < tot_vert-2; i++)
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
    mdp.graph->make_holding_time_constant_all();
    toc();
    
    mdp.write_pomdp_file();

    mdp.plot_trajectory();
    mdp.graph->plot_graph();
    bot_lcmgl_switch_buffer(lcmgl);

    //mdp.run_lqg();
    cout<<"Finished" << endl;

    return 0;
}
