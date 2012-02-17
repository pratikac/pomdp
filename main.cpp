#include "mdp.h"

int main(int argc, char** argv)
{
    int tot_vert = 20;
    float discount = 0.99;
    double process_noise = 0.08;
    if (argc > 1)
        tot_vert = atoi(argv[1]);
    
    discount = exp(-1/(float)tot_vert);
    
    if (argc > 2)
        discount = atof(argv[2]);
    if (argc > 3)
        process_noise = atof(argv[3]);
    cout<<"vert: "<< tot_vert << endl;
    cout<<"discount: "<< discount << endl;
    cout<<"process_noise: "<< process_noise << endl;

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
    //mdp.graph->make_holding_time_constant_all();
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
