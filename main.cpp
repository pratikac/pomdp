#if 0
#include "common.h"

int main()
{
    lcm_t *lcm = bot_lcm_get_global(NULL);
    bot_lcmgl_t *lcmgl = bot_lcmgl_init(lcm, "plotter");
    
    bot_lcmgl_color3f(lcmgl, 1, 0, 0);
    bot_lcmgl_point_size(lcmgl, 4.0);

    bot_lcmgl_begin(lcmgl, GL_LINES);
    
    for(int i=0; i< 100; i++)
    {
        bot_lcmgl_vertex3f(lcmgl, 0,0,0);
        bot_lcmgl_vertex3f(lcmgl, RANDF, RANDF, RANDF);
    }
    bot_lcmgl_end(lcmgl);

    bot_lcmgl_switch_buffer(lcmgl);

    return 0;
}

#endif

#include "mdp.h"

int main()
{
    lcm_t *lcm = bot_lcm_get_global(NULL);
    bot_lcmgl_t *lcmgl = bot_lcmgl_init(lcm, "plotter");
    bot_lcmgl_line_width(lcmgl, 2.0);
    
    System sys;
    Graph graph(sys, lcmgl);
    MDP mdp(graph, lcmgl);
    mdp.draw_lcm_grid();

    mdp.propagate_system();
    
    tic();
    mdp.graph->add_sample(true);
    cout<<"Start sampling" << endl;
    for(int i=0; i < 50; i++)
    {
        mdp.graph->add_sample(false);
        if(i % 100 == 0)
        {
            cout<<i<<" ";
            toc();
        }
        // mdp.graph->print_rrg();
        // cout<<"getchar: "<< endl; getchar();
    }
    mdp.graph->make_holding_time_constant_all();
    toc();
    
#if 1
    mdp.write_pomdp_file();
#endif

    mdp.plot_trajectory();
    mdp.graph->plot_graph();
    bot_lcmgl_switch_buffer(lcmgl);
    
    cout<<"Finished" << endl;

    return 0;
}
