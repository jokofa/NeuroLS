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

int main_old(int argc, char* argv[])
{
	double sols = 0;
	std::vector<std::vector<int>> routes; 

	//int n = 20;

		
		 
	// Read Problem or
	char input[VRPH_STRING_SIZE];
	string  input_filename;

	input_filename = string("C:/Users/inbox/Desktop/att48.vrp");
	strcpy_s(input, input_filename.c_str());
	int n = VRPGetDimension(input);

	VRP V(n);

	// Read Problem
	V.read_TSPLIB_file(input);
		
	// Generate Problem
	//V.load_problem(1, generate_nodes(n + 1), generate_demands(n), { -1.0, 50, -1.0, 0, 0 }, { {-1} }, VRPH_EXACT_2D, -1);

	ClarkeWright CW(n);
	CW.Construct(&V, 1.0, false);

	sols += V.get_total_route_length();
	cout << sols << "\n";
	routes = V.get_routes();

	//for (auto route : routes)
	//	for (auto node : route)
	//		cout << node<<", ";

	int *sol_buffer;
	sol_buffer = new int[n+1];
	V.export_solution_buff(sol_buffer);

	//for (int i = 0; i < n + 1; i++)
	//	cout << sol_buffer << ",";
	cout << "\n end";
// 
	//for (int route =0; route <= routes.size(); route++){
	//	for (int node =0; node <= routes[route].size(); node++){
	//		cout << " " << V.nodes[routes[route][node]].cummulative_distance <<"-";
	//	}
	//	
	//}
	
	

	return 0;
}
