#include "mdp.h"

/*
test cases:
1d
    n = 20, process_noise = 0.08
    n = 22, process_noise = 0.05
2d
    n = 64, process_noise = 0.01, obs = 25
    n = 100, process_noise = 0.01, obs = 25
    n = 196, process_noise = 0.01, obs = 25
*/
int main(int argc, char** argv)
{
    int tot_vert = 20;
    double process_noise = 0.08;
    if (argc > 1)
        tot_vert = atoi(argv[1]);
    
    double discount = pow(0.99, pow((float)tot_vert/log(tot_vert), 0.5));
    
    if (argc > 2)
        process_noise = atof(argv[2]);

    cout<<"vert: "<< tot_vert << endl;
    cout<<"process_noise: "<< process_noise << endl;
    cout<<"discount: "<< discount << endl;

    srand(0);
    System sys(discount, process_noise);
    sys.sample_control_observations(tot_vert);
    Graph graph(sys, tot_vert);
    MDP mdp(graph);
   
    tic();
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
    cout<<"Finished" << endl;

    return 0;
}
