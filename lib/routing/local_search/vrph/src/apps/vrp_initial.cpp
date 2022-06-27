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
    /// A main() routine to illustrate usage of Clarke Wright and 
    /// Sweep methods to construct an initial solution.
    ///

    int i,n;
    char infile[200];
    char outfile[200];

    if(argc < 5 || (strncmp(argv[1],"-help",5)==0 || strcmp(argv[1],"-h")==0 || strcmp(argv[1],"--h")==0))
    {        
        VRPH_version();
        
        fprintf(stderr,"Usage: %s -f vrp_file -m method [-c] [-l <lambda>] [-t sol_file] [-h <heuristic>]\n",argv[0]);
        fprintf(stderr,
            "\t method should be 0 for CW, 1 for Sweep\n"
            "\t If -c option is given, then all routes are cleaned up at the end\n"
            "\t\t by running intra-route improvements\n"
            "\t Optional -l lambda parameter can be given for CW (default 1.0)\n"
            "\t\t the value of the lambda is recommended to be within [0.5, 2.0]\n"
            "\t Optional -t write resulting solution to file\n"
            "\t Optional -h <heuristic> applies the specified heuristics (can be repeated)\n"
            "\t\t to use heuristics you need ot enable the -c option\n"
            "\t\t default is ONE_POINT_MOVE, TWO_POINT_MOVE, and TWO_OPT\n"
            "\t\t others available are OR_OPT, THREE_OPT, and CROSS_EXCHANGE\n"
            "\t\t Example: -h OR_OPT -h THREE_OPT -h TWO_OPT\n"
            "\t\t \n");
            exit(-1);
    }

    double lambda = 1.0;
    bool has_filename = false, clean_up = false, has_heuristics = false, write_outfile = false;
    int heuristics = 0;

    for(i=1;i<argc;i++)
    {
        if(strcmp(argv[i],"-f")==0 && (i + 1)<argc)
        {
            strcpy(infile,argv[i+1]);
            has_filename=true;            
        }

        if (strcmp(argv[i], "-t") == 0 && (i + 1)<argc)
        {
            strcpy(outfile, argv[i + 1]);
            write_outfile = true;
        }

        if (strcmp(argv[i], "-h") == 0 && (i + 1)<argc)
        {
            has_heuristics = true;
            if (strcmp(argv[i + 1], "ONE_POINT_MOVE") == 0)
                heuristics += ONE_POINT_MOVE;
            if (strcmp(argv[i + 1], "TWO_POINT_MOVE") == 0)
                heuristics += TWO_POINT_MOVE;
            if (strcmp(argv[i + 1], "TWO_OPT") == 0)
                heuristics += TWO_OPT;
            if (strcmp(argv[i + 1], "OR_OPT") == 0)
                heuristics += OR_OPT;
            if (strcmp(argv[i + 1], "THREE_OPT") == 0)
                heuristics += THREE_OPT;
            if (strcmp(argv[i + 1], "CROSS_EXCHANGE") == 0)
                heuristics += CROSS_EXCHANGE;
            if (strcmp(argv[i + 1], "THREE_POINT_MOVE") == 0)
                heuristics += THREE_POINT_MOVE;
        }
    }
    
    if (has_filename==false)
        report_error("No input file given\n");
    
    if (has_heuristics==false)
        heuristics = ONE_POINT_MOVE + TWO_POINT_MOVE + TWO_OPT;
    
    n=VRPGetDimension(infile);
    // Get # of non-VRPH_DEPOT nodes
    VRP V(n);

    // Now process the options
    int method=-1;
    for(i=1;i<argc;i++)
    {
         if (strcmp(argv[i], "-c") == 0)
            clean_up = true;
         if(strcmp(argv[i],"-m")==0 && (i + 1)<argc)
            method=atoi(argv[i+1]);
         if (strcmp(argv[i], "-l") == 0 && (i + 1)<argc)
             lambda = atof(argv[i + 1]);
    }
    if(method<0)
        report_error("method must be 0 or 1\n");

    // Load the problem data
    V.read_TSPLIB_file(infile);

    Sweep sweep;
    ClarkeWright CW(n);

    if(method==0)
    {
        // Use Clarke Wright with \lambda=1
        printf("Finding initial solution using Clarke-Wright algorithm\n");
        CW.Construct(&V, lambda, false);
        
    }
    else
    {
        printf("Finding initial solution using Sweep algorithm\n");
        sweep.Construct(&V);
    }

    if(clean_up)
    {
        printf("Total route length before clean up: %f\n",V.get_total_route_length()-V.get_total_service_time());
        V.normalize_route_numbers();
        for(i=1;i<=V.get_total_number_of_routes();i++)
            V.clean_route(i, heuristics);
    }

    // Print a summary of the solution
    V.summary();
    if (write_outfile)
    {
        //V.show_routes();
        V.write_solution_file(outfile);
    }
    
    return 0;

}

