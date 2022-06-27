
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "VRPH.h"


using namespace pybind11::literals;

namespace py = pybind11;

// VRP clarke_construct (VRP V){
//    CW = ClarkeWright(V.get_num_nodes());
//    CW(&V,1,false);
//    return(V)
//}


PYBIND11_MODULE(VRPH, m) {

    m.doc() = "VRPH port to python - ISMLL";

    //m.def("clarke_construct", &clarke_construct, "VRP_instance"_a);
    py::class_<CrossExchange>(m, "CrossExchange")
        .def("route_search", &CrossExchange::route_search, "VRP_instance"_a, "first_route"_a, "second_route"_a, "criteria"_a);

    py::class_<OnePointMove>(m, "OnePointMove")
        .def("route_search", &OnePointMove::route_search, "VRP_instance"_a, "first_route"_a, "second_route"_a, "rules"_a)
        .def("search", &OnePointMove::search, "VRP_instance"_a, "node"_a, "rules"_a);

    py::class_<OrOpt>(m, "OrOpt")
        .def("route_search", &OrOpt::route_search, "VRP_instance"_a, "first_route"_a, "second_route"_a, "k"_a, "rules"_a)
        .def("search", &OrOpt::search, "VRP_instance"_a, "node_a"_a, "node_b"_a, "rules"_a);

    py::class_<ThreeOpt>(m, "ThreeOpt")
        .def("search", &ThreeOpt::search, "VRP_instance"_a, "node"_a, "criteria"_a);

    py::class_<ThreePointMove>(m, "ThreePointMove")
        .def("route_search", &ThreePointMove::route_search, "VRP_instance"_a, "first_route"_a, "second_route"_a, "criteria"_a)
        .def("search", &ThreePointMove::search, "VRP_instance"_a, "node"_a, "criteria"_a);

    py::class_<TwoOpt>(m, "TwoOpt")
        .def("route_search", &TwoOpt::route_search, "VRP_instance"_a, "first_route"_a, "second_route"_a, "criteria"_a)
        .def("search", &TwoOpt::search, "VRP_instance"_a, "node"_a, "criteria"_a);

    py::class_<TwoPointMove>(m, "TwoPointMove")
        .def("route_search", &TwoPointMove::route_search, "VRP_instance"_a, "first_route"_a, "second_route"_a, "rules"_a)
        .def("search", &TwoPointMove::search, "VRP_instance"_a, "node"_a, "rules"_a);

    py::class_<ClarkeWright>(m, "ClarkeWright")
    .def(py::init <int>(), py::return_value_policy::reference)
    .def("Construct", &ClarkeWright::Construct, "VRP_instance"_a, "lambda"_a, "use_neighbor_list"_a)
    .def("create_initial", &ClarkeWright::create_initial, "VRP_instance"_a, "lambda"_a, "use_neighbor_list"_a)
    .def("CreateSavingsMatrix", &ClarkeWright::CreateSavingsMatrix, "VRP_instance"_a, "lambda"_a, "use_neighbor_list"_a);
    
    py::class_<Sweep>(m, "Sweep")
    .def(py::init(), py::return_value_policy::reference)
    .def("Construct", &Sweep::Construct, "VRP_instance"_a);

    
    py::class_<VRP>(m, "VRP")
        .def(py::init <int>(), py::return_value_policy::reference)
        .def(py::init <int, int>(), py::return_value_policy::reference)

        // TSPLIB file processing and display solutions
        .def("read_TSPLIB_file", &VRP::read_TSPLIB_file, "TSP file"_a)
        .def("load_problem", &VRP::load_problem, py::return_value_policy::reference)
        .def("show_next_array", &VRP::show_next_array, py::return_value_policy::reference)
        .def("show_pred_array", &VRP::show_pred_array, py::return_value_policy::reference)
        .def("verify_routes", &VRP::verify_routes)
        .def("check_fixed_edges", &VRP::check_fixed_edges)
        .def("create_pred_array", &VRP::create_pred_array)
        .def("print_stats", &VRP::print_stats, py::return_value_policy::reference)
        .def("get_perm", &VRP::get_perm, py::return_value_policy::reference)
        .def("show_routes", &VRP::show_routes, py::return_value_policy::reference)
        .def("show_route", &VRP::show_route, py::return_value_policy::reference)
        .def("summary", &VRP::summary, py::return_value_policy::reference)
        //.def("return_objective", &VRP::return_objective)


        // Read/write result operations
        .def("write_solution_file", &VRP::write_solution_file, py::return_value_policy::reference)
        .def("write_solutions", &VRP::write_solutions, py::return_value_policy::reference)
        .def("write_tex_file", &VRP::write_tex_file, py::return_value_policy::reference)
        .def("read_solution_file", &VRP::read_solution_file, py::return_value_policy::reference)
        .def("read_optimum_file", &VRP::read_optimum_file, py::return_value_policy::reference)
        .def("read_fixed_edges", &VRP::read_fixed_edges, py::return_value_policy::reference)
        .def("use_initial_solution", &VRP::use_initial_solution, "Set the initial solution manually for the model", "initial_solution"_a, py::return_value_policy::reference)


        .def("export_solution_buff", &VRP::export_solution_buff, py::return_value_policy::reference)
        .def("import_solution_buff", &VRP::import_solution_buff, py::return_value_policy::reference)
        .def("export_canonical_solution_buff", &VRP::export_canonical_solution_buff, py::return_value_policy::reference)


        .def("reset", &VRP::reset)
        .def("Single_solve", &VRP::Single_solve, py::return_value_policy::reference)
        .def("Single_node_solve", &VRP::Single_node_solve, py::return_value_policy::reference)
        .def("detailed_solve", &VRP::detailed_solve, "Attempts to apply the given local heuristics with the rules specified,\
                                the heuristics are presented in an integer form as per the identifiers.\
                                They are applied to a list of nodes or [-1] for all nodes.\
                                If all nodes are selected you can also select the first (steps) nodes to apply on or 0 for all.\
                                Will repeat a number of iterations or until convergance.", "operators"_a, "rule"_a, "nodes_to_operate"_a, "err_max"_a,"steps"_a, "iters"_a, "converge"_a, py::return_value_policy::reference)
        //.def("accept_solution", &VRP::accept_solve, "if true the current buffer is left as is, else the previous buffer is restored", "accept_bool"_a, py::return_value_policy::reference)
        .def("reject_move", &VRP::reject_move, "restores the previous solution buffer, effectively reversing the last move", py::return_value_policy::reference)

        //access private data
        .def("get_num_nodes", &VRP::get_num_nodes, py::return_value_policy::reference)
        .def("get_total_route_length", &VRP::get_total_route_length, py::return_value_policy::reference)
        .def("get_total_service_time", &VRP::get_total_service_time, py::return_value_policy::reference)
        .def("get_best_sol_buff", &VRP::get_best_sol_buff, py::return_value_policy::reference)
        .def("get_best_total_route_length", &VRP::get_best_total_route_length, py::return_value_policy::reference)
        .def("get_total_number_of_routes", &VRP::get_total_number_of_routes, py::return_value_policy::reference)
        .def("get_num_original_nodes", &VRP::get_num_original_nodes, py::return_value_policy::reference)
        .def("get_num_days", &VRP::get_num_days, py::return_value_policy::reference)
        .def("get_best_known", &VRP::get_best_known, py::return_value_policy::reference)
        .def("set_best_total_route_length", &VRP::set_best_total_route_length, py::return_value_policy::reference)
        .def("get_max_route_length", &VRP::get_max_route_length, py::return_value_policy::reference)
        .def("get_max_veh_capacity", &VRP::get_max_veh_capacity, py::return_value_policy::reference)
        .def("create_distance_matrix", &VRP::create_distance_matrix, py::return_value_policy::reference)
        .def("create_neighbor_lists", &VRP::create_neighbor_lists, py::return_value_policy::reference)
        .def("reverse_route", &VRP::reverse_route, py::return_value_policy::reference)
        //.def("get_nodes", &VRP::get_nodes)
        .def("get_route", &VRP::get_route, "route_number"_a, py::return_value_policy::copy)
        .def("set_random_seed", &VRP::set_random_seed, "seed"_a, py::return_value_policy::reference)
        .def("call_clarke", &VRP::call_clarke, py::return_value_policy::reference)
        .def("get_dummy_index", &VRP::get_dummy_index, py::return_value_policy::reference)
        //.def("update_dynamic_node_data", &VRP::update_dynamic_node_data, py::return_value_policy::reference)
        .def("get_routes", &VRP::get_routes, py::return_value_policy::copy)
        .def("set_max_num_routes", &VRP::set_max_num_routes, "Set the maximum number of routes, default:10,000", "max_routes"_a, py::return_value_policy::copy)
        .def("get_dynamic_node_data", &VRP::get_dynamic_node_data, py::return_value_policy::copy);

    m.def("VRPGetDimension", &VRPGetDimension, "Get the problem size from the instance file", "tsp_file"_a, py::return_value_policy::reference);
}