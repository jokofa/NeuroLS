#include "VRPH.h"

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>

using namespace std::chrono;
using namespace std;

int main(int argc, char* argv[])
{

    // create a file name and cast it from string to char array
    char input[VRPH_STRING_SIZE];
    string filename;

    //int initial_sol[101];
    std::vector<int>initial_sol(101);

    initial_sol[0] = -1;
    for (int sizes = 1; sizes <= 100; sizes++) {
        initial_sol[sizes] = sizes+1;
    }
    initial_sol[100] = 0;
    


    int heuristics = 0;
    int rules = VRPH_BEST_ACCEPT;
    heuristics |= (TWO_OPT);

    vector <float> times;
    vector <float> objectives;
    vector <float> initials;
    std::cout << "Beginning iterations";


    for (int instance = 0; instance < 100; instance++) {
        filename = string("../data/tsp100/");
        filename += string("tsp100_num_") + to_string(instance) + string("_seed1357.vrp");
        strcpy_s(input, filename.c_str());

        auto start = high_resolution_clock::now();
        // get the number of nodes in the VRP instance
        int n = VRPGetDimension(input);

        // initialize the VRP with n non-depot nodes and read data
        VRP V(n);
        V.read_TSPLIB_file(input);
        //V.use_initial_solution(initial_sol, n + 1);
        //initials.push_back(V.return_objective());

        //V.Single_solve(heuristics, rules, 0.000001, 100, false);
     
        auto stop = high_resolution_clock::now();
        duration <float> elapsed = (stop - start);

        times.push_back(elapsed.count());
        //objectives.push_back(V.return_objective());

    }

    /*
    float avg;
    float timeTotal = 0;
    float sumTotal = 0;
    for (int k = 0; k < objectives.size(); ++k) {
        sumTotal += objectives[k];
    }
    for (int k = 0; k < times.size(); ++k) {
        timeTotal += times[k];
    }
    avg = sumTotal / objectives.size();
    std::cout << "the average is: "<<avg << endl;
    std::cout << "the time is: " << timeTotal << endl;

    ofstream output("tsp100-twoopt.csv");
    int vsize = times.size();
    output << "time" << "," << "initial" << "," << "post" <<endl;
    for (int n = 0; n < vsize; n++)
    {
        output << times[n] << "," << initials[n] << "," << objectives[n] << endl;
    }
    output.close();
    */
    return 0;
}