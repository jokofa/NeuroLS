

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
int main(int argc, char* argv[])
{

    // initialize variables
    std::vector <double> demands = { -1.0 };
    double cap = -1.0;

    //setup an output file to record results
    string  output_filename = string("output-file");
    char output[VRPH_STRING_SIZE];
    strcpy_s(output, output_filename.c_str());
    std::vector<std::vector <double>>;

    int n;


    // create a file name and cast it from string to char array
    char input[VRPH_STRING_SIZE];
    string  input_filename;

    input_filename = string("C:/Users/inbox/Desktop/att48.vrp");
    strcpy_s(input, input_filename.c_str());

    // get the number of nodes in the VRP instance
    n = VRPGetDimension(input);

    VRP V(n+1);

    // load problem and create initial solution
    //V.load_problem(TYPE, coords, demands, { -1.0, cap, -1.0, 0, 0 }, VRPH_EXACT_2D, -1);

    // Read Problem
    V.read_TSPLIB_file(input);

    ClarkeWright CW(n);
    CW.Construct(&V, 1.2, false);

    //vector <vector <int>> initial_sol;
    //middle_sol = generateRandom(n);
    //middle_sol.push_back(0);
    //initial_sol = {{14, 3, 1, 79, 38, 108, 91},
    //    {30, 2, 80, 71},
    //    {52, 50, 68, 54, 41, 12, 58, 19},
    //    {10, 86, 73, 35, 57, 13, 23, 48},
    //    {17, 95, 96, 88, 75, 21, 72, 63},
    //    {76, 83, 22, 61, 60, 104, 64, 36, 59},
    //    {39, 28, 31, 27, 65, 99, 43, 93},
    //    {90, 16, 29, 78, 32, 105, 67, 89},
    //    {25, 103, 24, 4, 101, 9, 33, 102, 55, 56},
    //    {70, 44, 81, 94, 37, 7, 107, 82},
    //    {62, 100, 47, 40, 8, 49},
    //    {45, 34, 106, 42, 97, 92, 20, 74},
    //    {51, 85, 11, 98, 66, 53, 15, 26, 87},
    //    {6, 5, 46, 77, 84, 18, 69}};

    //V.use_initial_solution(initial_sol);


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


    vector <int> accept_types{ VRPH_LI_ACCEPT, VRPH_FIRST_ACCEPT, VRPH_BEST_ACCEPT };
    vector <string> acceptors{ "VRPH_LI_ACCEPT", "VRPH_FIRST_ACCEPT", "VRPH_BEST_ACCEPT" };

    vector <int> move_types{ TWO_OPT, THREE_OPT, ONE_POINT_MOVE, TWO_POINT_MOVE, OR_OPT, CROSS_EXCHANGE, THREE_POINT_MOVE };
    vector <string> movers{ "TWO_OPT", "THREE_OPT", "ONE_POINT_MOVE", "TWO_POINT_MOVE", "OR_OPT", "CROSS_EXCHANGE", "THREE_POINT_MOVE" };

    V.set_max_num_routes(3);

    for (int j = 0; j <= 4; j++) {

        int rules = accept_types[j] + VRPH_SAVINGS_ONLY + VRPH_FREE + VRPH_RANDOMIZED + VRPH_INTRA_ROUTE_ONLY;

        for (int i = 0; i <= 6; i++) {
            heuristics = move_types[i];

            cout << "\n Now testing: " << movers[i] << " in mode: " << acceptors[j] << "\n";

            V.detailed_solve(heuristics, rules, { 2 }, 0, 100, 0.001, false);

            current_route = V.get_total_route_length();
            best_route = V.get_best_total_route_length();
            //V.show_routes();
            printf("\n Current best: %f, Current: %f", best_route, current_route);
            printf("\n Number of routes = %u", V.get_total_number_of_routes());
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
