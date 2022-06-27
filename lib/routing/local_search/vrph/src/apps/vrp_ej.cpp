////////////////////////////////////////////////////////////
//                                                        //
// This file is part of the VRPH software package for     //
// generating solutions to vehicle routing problems.      //
// VRPH was developed by Chris Groer (cgroer@gmail.com).  //
//                                                        //
// (c) Copyright 2010 Chris Groer.                        //
// All Rights Reserved.  VRPH is licensed under the       //
// Common Public License.  See LICENSE file for details.  //
//                                                        //
////////////////////////////////////////////////////////////

#include "VRPH.h"

#define RANDOM 0
#define REGRET 1

int main(int argc, char *argv[])
{
    ///
    /// A main() routine to test the procedures and performance of 
    /// various routines for ejecting and injecting groups of nodes
    /// using two strategies.
    ///

    char infile[VRPH_STRING_SIZE];
    char solfile[VRPH_STRING_SIZE];
    char outfile[VRPH_STRING_SIZE];
    bool has_outfile=false, has_solfile=false, has_cutoff=false, new_lambdas=false;
    int i,j,n,num_ejected, num_trials, num_initial_solutions, num_improvements;
    int *ejected_buff, *heur_solbuff, *ej_solbuff, *best_solbuff;
    int method=-1;
    double cutoff_time = 0.0;
    bool has_heuristics=false;
    int heuristics=0;
    double lambda_vals[VRPH_MAX_NUM_LAMBDAS];
    
    
    clock_t start, stop, ticks_left, run_start;

    // Default values
    bool verbose=false;

    if(argc < 7 || (strncmp(argv[1],"-help",5)==0 || strcmp(argv[1],"-h")==0 || strcmp(argv[1],"--h")==0))
    {
        VRPH_version();
        
        fprintf(stderr,"Usage: %s -f <vrp_file> -j <num_ejected> -t <num_trials> -m <method> [options]\n",argv[0]);
        fprintf(stderr,
            "\t <num_ejected> should be something less than 20 or so\n"
            "\t <num_trials> can be fairly large as the procedure is fast (say 1000)\n"
            "\t <method> must be 0 (RANDOM search), 1 (REGRET search)\n"
            
            "\t Options:\n"

            "\t-help prints this help message\n"
            
            "\t-h <heuristic> applies the specified heuristics (can be repeated)\n"
            "\t\t default is ONE_POINT_MOVE, TWO_POINT_MOVE, and TWO_OPT\n"
            "\t\t others available are OR_OPT, THREE_OPT, and CROSS_EXCHANGE\n"
            "\t\t Example: -h OR_OPT -h THREE_OPT -h TWO_OPT -h ONE_POINT_MOVE\n"
            "\t\t Setting -h KITCHEN_SINK applies all heuristics in the \n"
            "\t\t improvement phase\n"
            
            "\t-l <num_lambdas> run the savings procedure for num_lambdas\n"
            "\t\t different initial solutions using a random lambda chosen\n"
            "\t\t from (0.5,2.0)\n"
            "\t\t if num_lambdas is not set, use lambdas {.6, 1.4, 1.6}.\n"
            
            "\t-sol <sol_file> begins with an existing solution contained\n"
            "\t\t in sol_file\n"
            
            "\t-v prints verbose output to stdout\n"
            
            "\t-cutoff <runtime> will stop after given runtime (in seconds)\n"
#if VRPH_ADD_ENTROPY
            "\t-r will search the neighborhood in a random fashion\n"

            "\t-s <seed> will set the random seed\n"
#endif
            "\t-out <out_file> writes the solution to the provided file\n");
            
            
        
        exit(-1);
    }

    bool has_filename=false;
    for(i=1;i<argc;i++)
    {
        if(strcmp(argv[i],"-f")==0)
        {
            strcpy(infile,argv[i+1]);
            has_filename=true;            
        }
    }

    if(has_filename==false)
        report_error("No input file given\n");

    n=VRPGetDimension(infile);
    // Get # of non-VRPH_DEPOT nodes
    VRP V(n);

    // Declare some buffers for solutions, nodes to eject, etc.
    ejected_buff=new int[n+2];
    heur_solbuff=new int[n+2];
    ej_solbuff=new int[n+2];
    best_solbuff=new int[n+2];
    num_ejected=num_trials=0;
    num_initial_solutions=1;

    // Now process the options
    for(i=1;i<argc;i++)
    {
        if(strcmp(argv[i],"-v")==0)
            verbose=true;

        if(strcmp(argv[i],"-j")==0)
        {
            num_ejected=atoi(argv[i+1]);
            if (num_ejected>n)
            {
                // JHR: This prevents infinite loop if num_ejected is greater than n
                num_ejected = n;
                fprintf(stderr, "\nWARNING: Can only eject as many points as the problem has dimension! num_ejected forced to %d.\n", num_ejected);
            }
        }

        if(strcmp(argv[i],"-l")==0)
        {
            new_lambdas=true;
            num_initial_solutions=atoi(argv[i+1]);
        }

        if(strcmp(argv[i],"-sol")==0)
        {
            has_solfile=true;
            num_initial_solutions=1;
            strcpy(solfile,argv[i+1]);
        }
        
        if(strcmp(argv[i],"-h")==0)
        {
            has_heuristics=true;
            if(strcmp(argv[i+1],"ONE_POINT_MOVE")==0)
                heuristics|=ONE_POINT_MOVE;
            if(strcmp(argv[i+1],"TWO_POINT_MOVE")==0)
                heuristics|=TWO_POINT_MOVE;
            if(strcmp(argv[i+1],"TWO_OPT")==0)
                heuristics|=TWO_OPT;
            if(strcmp(argv[i+1],"OR_OPT")==0)
                heuristics|=OR_OPT;
            if(strcmp(argv[i+1],"THREE_OPT")==0)
                heuristics|=THREE_OPT;
            if(strcmp(argv[i+1],"CROSS_EXCHANGE")==0)
                heuristics|=CROSS_EXCHANGE;
            if(strcmp(argv[i+1],"THREE_POINT_MOVE")==0)
                heuristics|=THREE_POINT_MOVE;
            if(strcmp(argv[i+1],"KITCHEN_SINK")==0)
                heuristics|=KITCHEN_SINK;
        }

        if(strcmp(argv[i],"-t")==0)
            num_trials=atoi(argv[i+1]);

        if(strcmp(argv[i],"-out")==0)
        {
            has_outfile=true;
            strcpy(outfile,argv[i+1]);
        }

        if(strcmp(argv[i],"-m")==0)
        {
            method=atoi(argv[i+1]);
            if(method!=RANDOM && method!=REGRET)
            {
                fprintf(stderr,"Method must be either 0 (RANDOM search) or 1 (REGRET search)\n");
                exit(-1);
            }
        }
        
        if(strcmp(argv[i],"-cutoff")==0)
        {
            has_cutoff=true;
            cutoff_time=atof(argv[i+1]);
        }
        
#if VRPH_ADD_ENTROPY
        if(strcmp(argv[i],"-s")==0)
        {
            // Set the RNG seed
            lcgseed(atoi(argv[i+1]));
        }

        if(strcmp(argv[i],"-s")==0 || strcmp(argv[i],"-r")==0)
        {
            // JHR: Force use of randomization (othervise setting the seed makes little sense)
            heuristics+=VRPH_RANDOMIZED;
        }
#endif
    }
    
    if(has_heuristics==false)
        // Use default set of operators
        heuristics|=(ONE_POINT_MOVE+TWO_POINT_MOVE+TWO_OPT);
    heuristics|=VRPH_USE_NEIGHBOR_LIST;
    
    // If the # of initial savings generated solutions is not set, use predefined lambdas
    if(new_lambdas==false)
    {
        // Use .6, 1.4, 1.6
        num_initial_solutions=3;
        lambda_vals[0]=.6;
        lambda_vals[1]=1.4;
        lambda_vals[2]=1.6;
    }
    // Generate lambdas here, because random seed may be set in option processing above
    else
    {
        if(num_initial_solutions>VRPH_MAX_NUM_LAMBDAS)
        {
            fprintf(stderr,"%d>VRPH_MAX_NUM_LAMBDAS\n",num_initial_solutions);
            exit(-1);
        }

        for(int j=0;j<num_initial_solutions;j++)
        {
            // Generate a random lambda
            lambda_vals[j] = 0.5 + 1.5*((double)lcgrand(1));
        }
    }

    // Load the problem data
    V.read_TSPLIB_file(infile);    
    ClarkeWright CW(n);
    double best_heur_sol=VRP_INFINITY;
    double best_final_sol=VRP_INFINITY;
    double heur_time=0,ej_time=0;
    double lambda;
    double *heur_sols=new double[num_initial_solutions];
    double *final_sols=new double[num_initial_solutions];

    run_start=clock();
    for(i=0;i<num_initial_solutions;i++)
    {
        // Check if we have time for another solution
        stop=clock();
        if (has_cutoff && ((double)(stop-run_start))/CLOCKS_PER_SEC>cutoff_time)
            break;
        
        V.reset();
        if (has_solfile)
        {
            // JHR: copied from  vrp_rtr.cpp
            V.read_solution_file(solfile);
            
            if(verbose)
            {
                printf("Read in solution:\n");
                V.show_routes();
                printf("loaded solution %d: %f\n",i,V.get_total_route_length()-V.get_total_service_time());
            }
        }
        // Generate a solution w/ RTR that should be a good starting point for the search
        else
        {
            start=clock();
            lambda=lambda_vals[i];
            // Start with a clean VRP object
            CW.Construct(&V, lambda, false);
            if(verbose)
                printf("CW solution %d[%5.3f]: %f\n",i,lambda,V.get_total_route_length()-V.get_total_service_time());
            
            stop=clock();   
            
            heur_time+=((double)(stop-start))/CLOCKS_PER_SEC;
        }
        
        start=clock();
        ticks_left=run_start+(int)(cutoff_time*CLOCKS_PER_SEC)-start;
        V.RTR_solve(heuristics,30,5,2,.01,30,VRPH_LI_PERTURB,
            VRPH_FIRST_ACCEPT,false, has_cutoff, ticks_left);
        stop=clock();
        heur_time+=((double)(stop-start))/CLOCKS_PER_SEC;
        
        heur_sols[i]=V.get_total_route_length()-V.get_total_service_time();
        if(verbose)
            printf("RTR solution %d: %f\n",i,V.get_total_route_length()-V.get_total_service_time());
        
        // Record the value of the first solution
        if(i==0 || V.get_total_route_length()-V.get_total_service_time() <= best_heur_sol)
        {
            best_heur_sol = V.get_total_route_length()-V.get_total_service_time();
            if(verbose)
                printf("Found best sol: %f\n",V.get_total_route_length()-V.get_total_service_time());
        }
        // Export this solution to ej_solbuff
        V.export_solution_buff(ej_solbuff);

        double heur_obj=V.get_total_route_length()-V.get_total_service_time();
        if(verbose)
            printf("Starting ejection routines with solution %f\n",heur_obj);

        num_improvements=0;
        double orig_obj=heur_obj;
        start=clock();
        for(j=0;j<num_trials;j++)
        {
            // Check if we have time for another trial
            stop=clock();
            if (has_cutoff && ((double)(stop-run_start))/CLOCKS_PER_SEC>cutoff_time)
                break;
                
            // Start with the best solution derived from this RTR run
            V.import_solution_buff(ej_solbuff);
            // Now pick random nodes to eject - start by finding a random non-VRPH_DEPOT node
            int r=VRPH_DEPOT;
            while(r==VRPH_DEPOT)
                r=(int)(lcgrand(11)*(n-1));

            // Eject a set of random nodes near node r
            V.eject_neighborhood(r,num_ejected,ejected_buff);

            if(method==REGRET)
            {
                // Inject them using "cheapest insertion with regret"
                V.inject_set(num_ejected, ejected_buff,VRPH_REGRET_SEARCH, 50);
                double regret_obj=V.get_total_route_length()-V.get_total_service_time();
                if(regret_obj<orig_obj)
                {
                    if(verbose)
                        printf("Attempt %04d:  REGRET improved original: %f<%f\n",j, regret_obj,orig_obj);
                    V.export_solution_buff(ej_solbuff);
                    orig_obj=regret_obj;
                    num_improvements++;
                }
            }

            if(method==RANDOM)
            {
                // Inject them again using a random order and cheapest insertion
                V.inject_set(num_ejected, ejected_buff,VRPH_RANDOM_SEARCH, 50);        
                double random_obj=V.get_total_route_length();
                if(random_obj<orig_obj)
                {
                   if(verbose)
                        printf("Attempt %04d:  RANDOM improved original: %f<%f\n",j, random_obj,orig_obj);
                    V.export_solution_buff(ej_solbuff);
                    orig_obj=random_obj;
                    num_improvements++;
                }
            }
        }
        // end j loop
        
        
        // Import the best solution we found
        V.import_solution_buff(ej_solbuff);
        if(V.get_total_route_length()-V.get_total_service_time()<best_final_sol)
        {
            best_final_sol=V.get_total_route_length()-V.get_total_service_time();
            V.export_solution_buff(best_solbuff);
        }
        final_sols[i]=V.get_total_route_length()-V.get_total_service_time();
        
        stop=clock();
        ej_time+=(double)(stop-start)/CLOCKS_PER_SEC;
    }

    // Output is
    // heur[i] ej[i]
    if(verbose)
        for(i=0;i<num_initial_solutions;i++)
            printf("%5.3f %5.3f\n",heur_sols[i], final_sols[i]);
    
    // Restore the best solution found
    V.import_solution_buff(best_solbuff);
    // output
    // k best_ej heur_time+ej_time
    printf("%d %5.3f %5.2f",V.get_total_number_of_routes(),
        V.get_total_route_length()-V.get_total_service_time(), heur_time+ej_time);
    if(V.get_best_known()>0 && V.get_best_known()<VRP_INFINITY)
        printf(" %1.3f\n",(V.get_total_route_length()-V.get_total_service_time())/V.get_best_known());
    else
        printf("\n");
    
    if(has_outfile)
        V.write_solution_file(outfile);


    // Clean up
    delete [] ejected_buff;
    delete [] heur_solbuff;
    delete [] ej_solbuff;
    delete [] best_solbuff;
    delete [] heur_sols;
    delete [] final_sols;

    return 0;
}
