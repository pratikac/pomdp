#include <iostream>
#include <fstream>
#include <string.h>
#include <stdio.h>
#include <vector>
using namespace std;

int main()
{
    char line[2000];
    FILE *sim_dat = fopen("singleint_sim.dat", "r");

    while ( fgets(line, 2000, sim_dat) != NULL)
    {
        char *token;
        token = strtok(line, ",");
        while(token != NULL)
        {
            cout<< token << endl;
            token = strtok(NULL, ",");
        }
    }
    fclose(sim_dat);
    return 0;
}
