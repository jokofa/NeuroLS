#include "VRPH.h"

#include <iostream>

using namespace std;

vector <vector <double>> generate_nodes(int num_nodes) {
	vector<vector <double>> nodes;
	srand(time(NULL));
	for (int i = 0; i < num_nodes; i++) {

		double x = ((double)rand() / (RAND_MAX));
		double y = ((double)rand() / (RAND_MAX));
		vector <double> buf = { x, y };
		nodes.push_back(buf);
	}
	return nodes;
}
vector <double> generate_demands(int num_nodes) {
	srand(time(NULL));
	vector <double> demands;
	demands.push_back(0);
	for (double i = 0; i < num_nodes; i++) {

		demands.push_back((rand() % 9) + 1);
	}
	return demands;
}


int main(int argc, char* argv[])
{
	double sols = 0;
	std::vector<std::vector<int>> routes; 

	//int n = 20;

		
		 
	// Read Problem or
	char input[VRPH_STRING_SIZE];
	string  input_filename;

	input_filename = string("/home/jukebox/Desktop/instances/att48_augmented/D1_R45.vrp");
	strcpy(input, input_filename.c_str());
	int n = VRPGetDimension(input);

	VRP V(n+1);

	// Read Problem
	V.read_TSPLIB_file(input);
		
	// Generate Problem
	//V.load_problem(1, generate_nodes(n + 1), generate_demands(n), { -1.0, 50, -1.0, 0, 0 }, { {-1} }, VRPH_EXACT_2D, -1);

	ClarkeWright CW(n+1);
	CW.Construct(&V, 1.2, false);

	sols += V.get_total_route_length();
	cout << sols << "\n";
	routes = V.get_routes();
	V.show_routes();
	//for (auto route : routes)
	//	for (auto node : route)
	//		cout << node<<", ";

	// int *sol_buffer;
	// sol_buffer = new int[n+1];
	// V.export_solution_buff(sol_buffer);

	// for (int i = 0; i < n + 1; i++)
	// 	cout << sol_buffer << ",";
	// cout << "\n end";
// 
	//for (int route =0; route <= routes.size(); route++){
	//	for (int node =0; node <= routes[route].size(); node++){
	//		cout << " " << V.nodes[routes[route][node]].cummulative_distance <<"-";
	//	}
	//	
	//}
	


	std::vector<std::vector<int>> routes_cp;
    std::vector<int> route_cp;

    int heuristics = 0;
   

    vector <int> accept_types{VRPH_LI_ACCEPT, VRPH_FIRST_ACCEPT, VRPH_BEST_ACCEPT};
    vector <string> acceptors{"VRPH_LI_ACCEPT", "VRPH_FIRST_ACCEPT", "VRPH_BEST_ACCEPT"};

    vector <int> move_types{TWO_OPT, THREE_OPT, ONE_POINT_MOVE, TWO_POINT_MOVE, OR_OPT, CROSS_EXCHANGE, THREE_POINT_MOVE};
    vector <string> movers{ "TWO_OPT", "THREE_OPT", "ONE_POINT_MOVE", "TWO_POINT_MOVE", "OR_OPT", "CROSS_EXCHANGE", "THREE_POINT_MOVE"};


    for (int j = 0; j <= 2; j++) {

        int rules = accept_types[j] + VRPH_SAVINGS_ONLY + VRPH_FREE + VRPH_RANDOMIZED+ VRPH_INTRA_ROUTE_ONLY;

        for (int i = 0; i <= 6; i++) {
            heuristics = move_types[i];

            cout << "\n Now testing: " << movers[i] <<   " in mode: " << acceptors[j] << "\n";

            V.Single_solve(heuristics, rules, 0.001, 100, false);

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

            
		}
	}

	return 0;

}
