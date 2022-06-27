

#include "VRPH.h"

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>

using namespace std::chrono;
using namespace std;


int getNum(vector<int>& v)
{

    // Size of the vector
    int n = v.size();

    // Generate a random number
    srand(time(NULL));

    // Make sure the number is within
    // the index range
    int index = rand() % n;

    // Get random number from the vector
    int num = v[index];

    // Remove the number from the vector
    swap(v[index], v[n - 1]);
    v.pop_back();

    // Return the removed number
    return num;
}

// Function to generate n non-repeating random numbers
vector <int> generateRandom(int n)
{
    vector<int> v(n);
    vector<int> final;

    // Fill the vector with the values
    // 1, 2, 3, ..., n
    for (int i = 0; i < n; i++)
        v[i] = i + 1;

    // While vector has elements
    // get a random number from the vector and print it
    while (v.size()) {
        final.push_back(getNum(v));
    }
    return final;
}



////////////////////////////////
// MAIN
////////////////////////////////
int main_old(int argc, char* argv[])
{
    // create a file name and cast it from string to char array
    char input[VRPH_STRING_SIZE];
    char output[VRPH_STRING_SIZE];

    // problem type (0=TSP, 1=CVRP)
    int TYPE = 1;

    string filename;

    filename = string("output-cvrp");
    strcpy(output, filename.c_str());

    std::vector<std::vector <double>> coords{ {0.867559907, 0.187194022},{0.287683587, 0.001288328},
        {0.712445246, 0.183457872},{0.051661943, 0.108348364},
        {0.397062613, 0.211801082},{0.980159753, 0.574336511},
        {0.975650606, 0.478742592},{0.935970111, 0.163359959},
        {0.888688482, 0.242849781},{0.143262046, 0.263978356},
        {0.704457614, 0.335738212},{0.77157949, 0.370136563},
        {0.606287039, 0.344498841},{0.594176759, 0.262174697},
        {0.817701623, 0.345658335},{0.786781757, 0.086189127},
        {0.980546251, 0.667869106},{0.961350167, 0.462509662},
        {0.996135017, 0.608477197},{0.760113373, 0.458773512},
        {0.577557331, 0.434037619},{0.786008761, 0.14300438},
        {0.669801598, 0.281113115},{0.210383922, 0.361891265},
        {0.554882762, 0.299149704},{0.086962123, 0.129605772},
        {0.973331616, 0.620845143},{0.971527957, 0.512883278},
        {0.409301726, 0.097397578},{0.947178562, 0.580520484},
        {0.972043288, 0.360860603},{0.418062355, 0.425792322},
        {0.827879413, 0.408786395},{0.593661427, 0.154341665},
        {0.002963154, 0.28549343},{0.933779954, 0.486859057},
        {1, 0.59196601},{0.952331873, 0.289100747},
        {0.448853388, 0.364467921},{0.807910332, 0.275057975},
        {0.642231384, 0.018036589},{0.246843597, 0.202138624},
        {0.937902602, 0.631151765},{0.967405308, 0.417289358},
        {0.001288328, 0.344756506},{0.8769647, 0.385596496},
        {0.667997939, 0.419737181},{0.389461479, 0.250193249} };

    std::vector <double> demands = { -1.0 };

    std::vector<std::vector<double>> tw{ {-1.0} };

    double cap = -1.0;

    if (TYPE == 0){
        filename = string("./data/att48.tsp");
        
    } else {
        filename = string("/home/jukebox/Desktop/instances/att48_augmented/D1_R45.vrp");

        demands = {
            0.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0,
        };

        cap = 15.0;


    }

    strcpy(input, filename.c_str());

    
    // get the number of nodes in the VRP instance
    //int n = VRPGetDimension(input);

    int n = VRPGetDimension(input);
    VRP V(n+1);
    // load problem and create initial solution
    if (TYPE == 0){
        int n = coords.size();
        VRP V(n);
        V.load_problem(TYPE, coords, demands, {-1.0, cap, -1.0, 0, 0}, tw, VRPH_EXACT_2D, -1);

        vector <int> middle_sol;
        middle_sol = generateRandom(n);
        middle_sol.push_back(0);
        vector <vector <int>> initial_sol{middle_sol};
        V.use_initial_solution(initial_sol);
    }
    else {
        
        V.read_TSPLIB_file(input);    //this works!
        //V.load_problem(TYPE, coords, demands, {-1.0, cap, -1.0, 1.0, 0}, tw, VRPH_EXACT_2D, VRPH_FUNCTION);


        ClarkeWright CW(n+1);
        CW.Construct(&V,1.2,false);
        //Sweep().Construct(&V);

    }

    //int *sol_buffer;
    //sol_buffer = new int[n+1];
    //V.export_solution_buff(sol_buffer);
    //for (int j = 0; j < n + 2 ; j++) {
    //    printf("% d ", sol_buffer[j]);
    //}
    //printf("\n");   

 
    //V.write_solution_file(output);
    vector<vector<int>> rts;
    rts = V.get_routes();
    std::cout << "Reading Complete";
    
    V.show_routes();


    double current_route = V.get_total_route_length();
    double best_route = V.get_best_total_route_length();
    printf("\n Current best: %f, Current: %f", best_route, current_route);

    std::vector<std::vector<int>> routes_cp;
    std::vector<int> route_cp;

    int heuristics = 0;
   

    vector <int> accept_types{VRPH_LI_ACCEPT, VRPH_FIRST_ACCEPT, VRPH_BEST_ACCEPT};
    vector <string> acceptors{"VRPH_LI_ACCEPT", "VRPH_FIRST_ACCEPT", "VRPH_BEST_ACCEPT"};

    vector <int> move_types{TWO_OPT, THREE_OPT, ONE_POINT_MOVE, TWO_POINT_MOVE, OR_OPT, CROSS_EXCHANGE, THREE_POINT_MOVE};
    vector <string> movers{ "TWO_OPT", "THREE_OPT", "ONE_POINT_MOVE", "TWO_POINT_MOVE", "OR_OPT", "CROSS_EXCHANGE", "THREE_POINT_MOVE"};

    std::vector<std::vector<int>> routes;

    for (int j = 0; j <= 2; j++) {

        int rules = accept_types[j] + VRPH_SAVINGS_ONLY + VRPH_FREE + VRPH_RANDOMIZED+ VRPH_INTRA_ROUTE_ONLY;

        for (int i = 0; i <= 6; i++) {
            heuristics = move_types[i];

            cout << "\n Now testing: " << movers[i] <<   " in mode: " << acceptors[j] << "\n";

            V.detailed_solve(heuristics, rules, {2}, 0, 100, 0.001, false);

            current_route = V.get_total_route_length();
            best_route = V.get_best_total_route_length();
            V.show_routes();
            //printf("\n Current best: %f, Current: %f", best_route, current_route);
            V.update_dynamic_node_data();
            
            routes = V.get_routes();

            cout << "\n Cumm. Distances: "<< routes.size();

            for (int i =0; i < routes.size(); i++){
                for (int j =0; j < routes[i].size(); j++){
                    cout << " " << V.nodes[routes[i][j]].cummulative_demand<<"-";
                }
                
            }

            cout << "\n Cumm. Backwards Distances: "<< routes.size();

            for (int i =0; i < routes.size(); i++){
                for (int j =0; j < routes[i].size(); j++){
                    cout << " " << V.nodes[routes[i][j]].cummulative_demand_back<<"-";
                }
                
            }
	
        }
        //heuristics = THREE_OPT;

        //V.detailed_solve(heuristics, rules, { 1 }, 0.000001, 0, 1, false);
        // 
        //current_route = V.get_total_route_length();
        //best_route = V.get_best_total_route_length();

        //printf("\n Current best: %f, Current: %f", best_route, current_route);
        ////V.Single_solve(heuristics, rules, 0.000001, 100, false);


    }

	return 0;
}
