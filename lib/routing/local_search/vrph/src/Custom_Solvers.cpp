
//	Custom Solver

//#include <list>
//#include <algorithm>
//#include <iterator>
#include <iostream>
#include <vector>
#include <stdexcept>

#include "VRPH.h"


double VRP::Single_solve(int local_operators,  int rule, double err_max = 0.001, int iters = 1, bool converge = false)
{

    int random;
    if (local_operators & VRPH_RANDOMIZED)
        random = VRPH_RANDOMIZED;
    else
        random = 0;

    int operation_mode = VRPH_BEST_ACCEPT;


    if (rule & VRPH_FIRST_ACCEPT)
        operation_mode = VRPH_FIRST_ACCEPT;
    if (rule & VRPH_BEST_ACCEPT)
        operation_mode = VRPH_BEST_ACCEPT;
    if (rule & VRPH_LI_ACCEPT)
        operation_mode = VRPH_LI_ACCEPT;



    // Define the heuristics we may use

    OnePointMove OPM;
    TwoPointMove TPM;
    TwoOpt         TO;
    OrOpt         OR;
    ThreeOpt     ThreeO;
    CrossExchange    CE;
    ThreePointMove ThreePM;

    int i, j, rules;

    //this is the objective we have in mind
    int objective = VRPH_SAVINGS_ONLY;
    int move_accept = 0;

    if (rule & VRPH_SIMULATED_ANNEALING)
        move_accept = VRPH_SIMULATED_ANNEALING;
    if (rule & VRPH_RECORD_TO_RECORD)
        move_accept = VRPH_RECORD_TO_RECORD;
    if (rule & VRPH_DOWNHILL)
        move_accept = VRPH_DOWNHILL;
    if (rule & VRPH_FREE)
        //move_accept = VRPH_FREE + VRPH_INTRA_ROUTE_ONLY;
        move_accept = VRPH_FREE;

    int* perm = this->get_perm();


    // if we have a current solution, store it in buffer
    //this->export_solution_buff(this->best_sol_buff);
    this->capture_best_solution();

    // We will use this to compare heuristic answers
    normalize_route_numbers();


    // here we initialize variables
    double worst_obj = 0;
    //time_t start = clock();
    //time_t stop = 0;




    // each used in inter or intra heauristics
    int n = num_nodes;
    int R = total_number_of_routes;
    //int printers = 0;
    double final_obj;


    while (true) {

        if (!converge && iters == 0)
            break;
        iters--;
        if (iters == 0)
            break;

        final_obj = this->get_total_route_length();

        //check which heuristic to use and apply it
        if (local_operators & THREE_OPT)
        {
            //VRPH_FREE bipasses default meta-heuristic solution checks
            rules = VRPH_SAVINGS_ONLY + move_accept + operation_mode + objective;

            for (i = 1; i <= R; i++)
            {
                ThreeO.search(this, i, rules);
                if (this->total_route_length > worst_obj)
                    worst_obj = this->total_route_length;
            }
        }

        if (local_operators & ONE_POINT_MOVE)
        {
            rules = VRPH_SAVINGS_ONLY + move_accept + objective;
            if (random)
                random_permutation(perm, this->num_nodes);

            for (i = 1;i <= n;i++)
            {

                OPM.search(this, perm[i - 1], rules);
                if (this->total_route_length > worst_obj)
                    worst_obj = this->total_route_length;
            }
        }



        if (local_operators & TWO_POINT_MOVE)
        {
            rules = VRPH_SAVINGS_ONLY + move_accept + objective;
            if (random)
                random_permutation(perm, this->num_nodes);

            for (i = 1;i <= n;i++)
            {
                TPM.search(this, perm[i - 1], rules);
                if (this->total_route_length > worst_obj)
                    worst_obj = this->total_route_length;
            }
        }



        if (local_operators & TWO_OPT)
        {

            rules = VRPH_SAVINGS_ONLY + move_accept + objective + operation_mode;
            if (random)
                random_permutation(perm, this->num_nodes);

            for (i = 1;i <= n;i++)
            {
                TO.search(this, perm[i - 1], rules);
                if (this->total_route_length > worst_obj)
                    worst_obj = this->total_route_length;
            }
        }

        if (local_operators & THREE_POINT_MOVE)
        {
            rules = VRPH_SAVINGS_ONLY + move_accept + VRPH_INTER_ROUTE_ONLY + objective;
            if (random)
                random_permutation(perm, this->num_nodes);


            for (i = 1;i <= n;i++)
            {
                ThreePM.search(this, perm[i - 1], rules);
                if (this->total_route_length > worst_obj)
                    worst_obj = this->total_route_length;
            }
        }

        if (local_operators & OR_OPT)
        {
            rules = VRPH_SAVINGS_ONLY + move_accept + objective;
            if (random)
                random_permutation(perm, this->num_nodes);

            for (i = 1;i <= n;i++)
            {
                OR.search(this, perm[i - 1], 3, rules);
                if (this->total_route_length > worst_obj)
                    worst_obj = this->total_route_length;
            }

            for (i = 1;i <= n;i++)
            {
                OR.search(this, perm[i - 1], 2, rules);
                if (this->total_route_length > worst_obj)
                    worst_obj = this->total_route_length;
            }
        }

        if (local_operators & CROSS_EXCHANGE)
        {
            normalize_route_numbers();
            this->find_neighboring_routes();
            R = total_number_of_routes;
            rules = VRPH_SAVINGS_ONLY + move_accept + objective;
            if (random)
                random_permutation(perm, this->num_nodes);


            for (i = 1; i <= R - 1; i++)
            {
                for (j = 0;j <= 1;j++)
                {
                    CE.route_search(this, i, route[i].neighboring_routes[j], rules);
                    if (this->total_route_length > worst_obj)
                        worst_obj = this->total_route_length;
                }
            }
        }

        if (converge && (this->total_route_length < this->best_total_route_length) &&
            (VRPH_ABS(this->total_route_length - this->best_total_route_length) > err_max))
            break;
    }

    delete[] perm;

    if (operation_mode != VRPH_LI_ACCEPT) {
        this->capture_best_solution();
        return this->best_total_route_length;
    }
    return total_route_length;

}

double VRP::Single_node_solve(int node, int local_operators, int rule, double err_max = 0.001, int iters = 1, bool converge = false)
{

    int random;
    if (local_operators & VRPH_RANDOMIZED)
        random = VRPH_RANDOMIZED;
    else
        random = 0;

    int operation_mode = VRPH_BEST_ACCEPT;


    if (rule & VRPH_FIRST_ACCEPT)
        operation_mode = VRPH_FIRST_ACCEPT;
    if (rule & VRPH_BEST_ACCEPT)
        operation_mode = VRPH_BEST_ACCEPT;
    if (rule & VRPH_LI_ACCEPT)
        operation_mode = VRPH_LI_ACCEPT;



    // Define the heuristics we may use

    OnePointMove OPM;
    TwoPointMove TPM;
    TwoOpt         TO;
    OrOpt         OR;
    ThreeOpt     ThreeO;
    CrossExchange    CE;
    ThreePointMove ThreePM;

    int i, rules;

    //this is the objective we have in mind
    int objective = VRPH_SAVINGS_ONLY;
    int move_accept = 0;

    if (rule & VRPH_SIMULATED_ANNEALING)
        move_accept = VRPH_SIMULATED_ANNEALING;
    if (rule & VRPH_RECORD_TO_RECORD)
        move_accept = VRPH_RECORD_TO_RECORD;
    if (rule & VRPH_DOWNHILL)
        move_accept = VRPH_DOWNHILL;
    if (rule & VRPH_FREE)
        move_accept = VRPH_FREE + VRPH_INTRA_ROUTE_ONLY;

    int* perm = this->get_perm();

    // if we have a current solution, store it in buffer
    //this->export_solution_buff(this->best_sol_buff);
    this->capture_best_solution();

    // We will use this to compare heuristic answers
    normalize_route_numbers();


    // here we initialize variables
    double worst_obj = 0;
    //time_t start = clock();
    //time_t stop = 0;




    // each used in inter or intra heauristics
    //int n = num_nodes;
    int R = total_number_of_routes;
    //int printers = 0;
    double final_obj;


    while (true) {

        if (!converge && iters == 0)
            break;
        iters--;
        if (iters == 0)
            break;

        final_obj = this->get_total_route_length();

        //check which heuristic to use and apply it
        if (local_operators & THREE_OPT)
        {
            //VRPH_FREE bipasses default meta-heuristic solution checks
            rules = VRPH_SAVINGS_ONLY + move_accept + operation_mode + objective;

            for (i = 1; i <= R; i++)
            {
                ThreeO.search(this, i, rules);
                if (this->total_route_length > worst_obj)
                    worst_obj = this->total_route_length;
            }
        }

        if (local_operators & ONE_POINT_MOVE)
        {
            rules = VRPH_SAVINGS_ONLY + move_accept + objective;
            if (random)
                random_permutation(perm, this->num_nodes);


            OPM.search(this, node, rules);
            if (this->total_route_length > worst_obj)
                worst_obj = this->total_route_length;

        }



        if (local_operators & TWO_POINT_MOVE)
        {
            rules = VRPH_SAVINGS_ONLY + move_accept + objective;

            TPM.search(this, node, rules);
            if (this->total_route_length > worst_obj)
                worst_obj = this->total_route_length;

        }



        if (local_operators & TWO_OPT)
        {

            rules = VRPH_SAVINGS_ONLY + move_accept + objective + operation_mode;

            TO.search(this, node, rules);
            if (this->total_route_length > worst_obj)
                worst_obj = this->total_route_length;
            
        }

        if (local_operators & THREE_POINT_MOVE)
        {
            rules = VRPH_SAVINGS_ONLY + move_accept + VRPH_INTER_ROUTE_ONLY + objective;

                ThreePM.search(this, node, rules);
                if (this->total_route_length > worst_obj)
                    worst_obj = this->total_route_length;
            
        }

        if (local_operators & OR_OPT)
        {
            rules = VRPH_SAVINGS_ONLY + move_accept + objective;

            OR.search(this,node, 3, rules);
            if (this->total_route_length > worst_obj)
                worst_obj = this->total_route_length;
            
            OR.search(this, node, 2, rules);
            if (this->total_route_length > worst_obj)
                worst_obj = this->total_route_length;
            
        }

        if (local_operators & CROSS_EXCHANGE)
        {
            normalize_route_numbers();
            this->find_neighboring_routes();
            R = total_number_of_routes;
            rules = VRPH_SAVINGS_ONLY + move_accept + objective;

            for (i = 1; i <= R - 1; i++)
            {
                CE.route_search(this, i, route[i].neighboring_routes[node], rules);
                if (this->total_route_length > worst_obj)
                    worst_obj = this->total_route_length;
                
            }
        }

        if (converge && (this->total_route_length < this->best_total_route_length) &&
            (VRPH_ABS(this->total_route_length - this->best_total_route_length) > err_max))
            break;
    }

    delete[] perm;

    this->capture_best_solution();

    return this->best_total_route_length;


}

double VRP::detailed_solve(int local_operators, int rule, std::vector<int> node, int steps = 0, int iters = 1, double err_max = 0.001, bool converge = false) {
    //
    //define the rules of the solve
    //
   
    //define whether to randomize the node order when performing the operators
    int random;
    if (rule & VRPH_RANDOMIZED){
        random = 1;
    }
    else
        random = 0;

    //define the acceptance criteria
    //default
    int accept_mode = VRPH_BEST_ACCEPT;

    //accept regardless of better or worse
    if (rule & VRPH_FIRST_ACCEPT)
        accept_mode = VRPH_FIRST_ACCEPT;
    // run through all of the operations then return the best 
    if (rule & VRPH_BEST_ACCEPT)
        accept_mode = VRPH_BEST_ACCEPT;
    //stop at first better solution
    if (rule & VRPH_LI_ACCEPT)
        accept_mode = VRPH_LI_ACCEPT;

    //define the local operators
    //
    OnePointMove OPM;
    TwoPointMove TPM;
    TwoOpt         TO;
    OrOpt         OR;
    ThreeOpt     ThreeO;
    CrossExchange    CE;
    ThreePointMove ThreePM;

    int rules, k, m, i, j;

    //at the moment we focus on cost savings only
    int objective = VRPH_SAVINGS_ONLY;

    //criteria for move acceptance
    int search_mode = 0;

    if (rule & VRPH_SIMULATED_ANNEALING)
        search_mode = VRPH_SIMULATED_ANNEALING;
    if (rule & VRPH_RECORD_TO_RECORD)
        search_mode = VRPH_RECORD_TO_RECORD;
    if (rule & VRPH_DOWNHILL)
        search_mode = VRPH_DOWNHILL;
    if (rule & VRPH_FREE)
        //move_accept = VRPH_FREE + VRPH_INTRA_ROUTE_ONLY;
        search_mode = VRPH_FREE;

    if (rule & VRPH_USE_NEIGHBOR_LIST)
        search_mode += VRPH_USE_NEIGHBOR_LIST;
    //
    //define required variables
    //
    //get node permutation order
    //int* perm = this->get_perm();
    int* nodes_to_operate;
    nodes_to_operate = new int[num_nodes];

    /*
    printf("print perm\n");
    for (int i = 0; i <= sizeof perm; i++){
        std::cout << perm[i] << ", ";
    }
    */

    int* nodes_to_operate_temp;
    nodes_to_operate_temp = new int[num_nodes];

    //here we initialize variables for storing specific objectives
    double worst_obj = 0;
    //local variables to store current solution properties 
    int R = total_number_of_routes;

    //save current solution for reversion in case of not accepted move
    this->previous_total_route_length = this->get_total_route_length();
    this->export_solution_buff(this->previous_sol_buff);

    //
    //start the solve
    //we have two main options to be wary of: operations on a specific node (or node list) or solve for a number of nodes
    //
    
    //if we selected a node or node list then we operate on them only
    int perm_node_count;

    if (node[0] == -1)
        throw std::invalid_argument( "received negative node idx" );


    //printf("print provided nodes\n");


    perm_node_count = int(node.size());
    for (i = 0; i < perm_node_count; i++) {
        nodes_to_operate_temp[i] = node[i];

        //std::cout << node[i] << ", ";

    }
    if (steps > 0) {
        steps = unsigned(steps);
        if (node.size() < steps) {
            printf("\n Solve called with n_steps > list of nodes set: will use max node list size \n");
            steps = unsigned(node.size());
        }
        perm_node_count = steps;
        for (i = 0; i < perm_node_count; i++) 
            nodes_to_operate[i] = nodes_to_operate_temp[i];
    } else {
        nodes_to_operate = nodes_to_operate_temp;
    }
    
    //if we selected a specific number of steps then we operate on them only
    //could be node 0 to node 0+steps or some random step-size window if random permute
    /*
    else if (steps != 0) {
        perm_node_count = steps;
        for (int i = 0; i < perm_node_count; i++) {
            nodes_to_operate[i] = perm[i];
        }
    }
    //if we haven't selected anything then we just run through all nodes
    else{
        nodes_to_operate = perm;
        perm_node_count = this->num_nodes;
    }
    */

    //store the previous solution to be able to restore it on reject_move()
    this->previous_total_route_length = this->get_total_route_length();
    this->export_solution_buff(this->previous_sol_buff);


    // since we were overriding KITCHEN_SINK with TABU, we need to make sure neither is used
    if (rule & KITCHEN_SINK || rule & VRPH_TABU || local_operators & KITCHEN_SINK || local_operators & VRPH_TABU)
        throw std::invalid_argument( "<KITCHEN_SINK> or <VRPH_TABU> cannot be used with custom solver." );


    while (true) {

        //repeat the solve till we finish the number of iterations
        if (iters == 0) {
            break;
        }
        iters--;
        

        //PERTURBATION operators
        //Perturbation is done before any other possible local operator is applied
        if (local_operators & PERTURB_LI){
#if SOLVER_DEBUG
    printf("\nrun PERTURB_LI...");
#endif
            perturb();
            if (this->total_route_length > worst_obj)
                worst_obj = this->total_route_length;
        }


        //perturb at least 4 nodes or 20% of nodes (which ever is higher) and maximally 50% of them
        k = VRPH_MIN(VRPH_MAX(perm_node_count, VRPH_MAX((num_nodes+1)/5, 4)), (num_nodes+1)/2);

        if (local_operators & PERTURB_OSMAN){
#if SOLVER_DEBUG
    printf("\nrun PERTURB_OSMAN...");
#endif
            //args: (num_nodes, alpha)
            osman_perturb(k, .5+lcgrand(20));
            if (this->total_route_length > worst_obj)
                worst_obj = this->total_route_length;
        }
        
        if (local_operators & PERTURB_TARGET){
#if SOLVER_DEBUG
    printf("\nrun PERTURB_TARGET...");
#endif
            //custom targeted perturbation via osman_insert
            if (random && perm_node_count > 1)
                random_permutation(nodes_to_operate, perm_node_count);
            
            //std::cout << "num_nodes:" << num_nodes << std::endl;
            //std::cout << "perm_node_count:" << perm_node_count << std::endl;
            //std::cout << "k:" << k << std::endl;

            if (k > perm_node_count){
                if (VRPH_USE_NEIGHBOR_LIST==1){
                    // fill up node list with neighbors
                    //printf("neighbors");
                    nodes_to_operate_temp = nodes_to_operate;
                    m = perm_node_count;
                    i = 0;
                    while (m < k){
                        j = 0;
                        while (j < neighbor_list_size){
                            nodes_to_operate_temp[m] = nodes[nodes_to_operate[i]].neighbor_list[j].position;
                            m++;
                            if (m == k)
                                break;
                            j++;
                        }
                        i++;
                    }
                    
                } else
                throw std::invalid_argument( "trying to use <PERTURB_TARGET> with too small number of provided nodes and without neighborhood." );

            } else
                // use all provided nodes
                nodes_to_operate_temp = nodes_to_operate;

            //args: (num_nodes, nodelist, alpha)
            targeted_osman_perturb(k, nodes_to_operate_temp, .5+lcgrand(20));
            if (this->total_route_length > worst_obj)
                worst_obj = this->total_route_length;
        }


        //start looping through the local operators
        //these are node based methods

        //ONE POINT MOVE
        if (local_operators & ONE_POINT_MOVE)
        {
#if SOLVER_DEBUG
    printf("\nrun ONE_POINT_MOVE...");
#endif
            //concatenate the rules set above
            rules = accept_mode + search_mode + objective;
            //check whether a node permutation is required
            if (random)
                random_permutation(nodes_to_operate, perm_node_count);
            //loop through the nodes in the above list
            for (i = 1; i <= perm_node_count; i++)
            {
                OPM.search(this, nodes_to_operate[i - 1], rules);
                if (this->total_route_length > worst_obj)
                    worst_obj = this->total_route_length;
            }
        }

        //TWO POINT MOVE
        if (local_operators & TWO_POINT_MOVE)
        {
#if SOLVER_DEBUG
    printf("\nrun TWO_POINT_MOVE...");
#endif
            rules = accept_mode + search_mode + objective;
            if (random)
                random_permutation(nodes_to_operate, perm_node_count);

            for (i = 1; i <= perm_node_count; i++)
            {
                TPM.search(this, nodes_to_operate[i - 1], rules);
                if (this->total_route_length > worst_obj)
                    worst_obj = this->total_route_length;
            }
        }

        //TWO OPT
        if (local_operators & TWO_OPT)
        {
#if SOLVER_DEBUG
    printf("\nrun TWO_OPT...");
#endif
            rules = accept_mode + search_mode + objective;
            if (random)
                random_permutation(nodes_to_operate, perm_node_count);
            //printf("\n \n nodes are: ");
            for (i = 1; i <= perm_node_count; i++)
            {   
                //printf("%d-", nodes_to_operate[i - 1]);
                TO.search(this, nodes_to_operate[i - 1], rules);
                if (this->total_route_length > worst_obj)
                    worst_obj = this->total_route_length;
            }
           // printf("\n \n");
        }

        //THREE POINT MOVE
        if (local_operators & THREE_POINT_MOVE)
        {
#if SOLVER_DEBUG
    printf("\nrun THREE_POINT_MOVE...");
#endif
            rules = accept_mode + search_mode + VRPH_INTER_ROUTE_ONLY + objective;
            if (random)
                random_permutation(nodes_to_operate, perm_node_count);


            for (i = 1; i <= perm_node_count; i++)
            {
                ThreePM.search(this, nodes_to_operate[i - 1], rules);
                if (this->total_route_length > worst_obj)
                    worst_obj = this->total_route_length;
            }
        }

        //OR OPT
        if (local_operators & OR_OPT)
        {
#if SOLVER_DEBUG
    printf("\nrun OR_OPT...");
#endif
            rules = accept_mode + search_mode + objective;
            if (random)
                random_permutation(nodes_to_operate, perm_node_count);

            for (i = 1; i <= perm_node_count; i++)
            {
                OR.search(this, nodes_to_operate[i - 1], 3, rules);
                if (this->total_route_length > worst_obj)
                    worst_obj = this->total_route_length;
            }

            for (i = 1; i <= perm_node_count; i++)
            {
                OR.search(this, nodes_to_operate[i - 1], 2, rules);
                if (this->total_route_length > worst_obj)
                    worst_obj = this->total_route_length;
            }
        }

        //following are route based methods

        // THREE OPT
        if (local_operators & THREE_OPT)
        {
#if SOLVER_DEBUG
    printf("\nrun THREE_OPT...");
#endif
            //VRPH_FREE bipasses default meta-heuristic solution checks
            rules = accept_mode + search_mode + objective;
            
            for (i = 1; i <= R; i++)
            {   
                try {
                    ThreeO.search(this, i, rules);
                } catch (const std::length_error& e) {
                    std::cerr << "\nWARNING: " << e.what() << std::endl;
                    continue;
                }

                if (this->total_route_length > worst_obj)
                    worst_obj = this->total_route_length;
            }
        }

        // CROSS EXCHANGE
        if (local_operators & CROSS_EXCHANGE)
        {
#if SOLVER_DEBUG
    printf("\nrun CROSS_EXCHANGE...");
#endif
            // cross exchange works only for R>1 routes
            // -> makes no sense for TSP and will not have any effect
            if (problem_type == VRPH_TSP)
                throw std::invalid_argument( "trying to use <CROSS EXCHANGE> operator for TSP." );

            normalize_route_numbers();
            this->find_neighboring_routes();
            R = total_number_of_routes;
            rules = accept_mode + search_mode + objective;

            for (i = 1; i <= R - 1; i++)
            {
                for (int j = 0; j <= 1; j++)
                {
                    CE.route_search(this, i, route[i].neighboring_routes[j], rules);
                    if (this->total_route_length > worst_obj)
                        worst_obj = this->total_route_length;
                }
            }
        }

        //check if converged
        if (converge && (VRPH_ABS(this->total_route_length - this->best_total_route_length) > err_max))
            break;

    }

    //release some variables
    //delete[] perm;
    //delete[] node;

    //if this is the best solution store it, add to warehouse if there is room
    this->capture_best_solution();

    ////if we want to work the next step on the best solution, set it as the current one
    ////TODO: look at just copying the best sol buffer as it could be faster
    //if (accept_mode != VRPH_FIRST_ACCEPT) {
    //    //take the best solution and put it in the current buffer
    //    //this should automatically set the current length as well
    //    this-> import_solution_buff(this->best_sol_buff);

    //    //take the best length and set it as the current length
    //    //total_route_length = best_total_route_length;
    //}

    //return current
    return this->get_total_route_length();
}


