/* SAS modified this file. */
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
//#include <iostream>

//https://wiki.sei.cmu.edu/confluence/display/c/ERR34-C.+Detect+errors+when+converting+a+string+to+a+number
#define myatoi(si,buff){                                                \
      errno = 0;                                                        \
      char * end;                                                       \
      const long sl = strtol(buff,&end,10);                           \
      if(end == buff){                                                \
         report_error("%s not a decimal number\n",__FUNCTION__);        \
      }                                                                 \
      else if('\0' != *end){                                            \
         report_error("%s extra characters at end of input\n",__FUNCTION__); \
      }                                                                 \
      else if((LONG_MIN==sl || LONG_MAX==sl) && ERANGE==errno){         \
         report_error("%s out of range type long\n",__FUNCTION__);      \
      }                                                                 \
      else if(sl > INT_MAX){                                            \
         report_error("%s greater than INT_MAX\n",__FUNCTION__);        \
      }                                                                 \
      else if(sl < INT_MIN){                                            \
         report_error("%s less than INT_MIN\n",__FUNCTION__);           \
      }                                                                 \
      else                                                              \
         (si) = (int)sl;                                                \
   }


void VRP::read_solution_file(const char* filename)
{
    ///
    /// Imports a solution from filename.  File is assumed to be in the
    /// format produced by VRP.write_solution_file
    ///

    FILE* in;

    if ((in = fopen(filename, "r")) == NULL)
    {
        fprintf(stderr, "Error opening %s for reading\n", filename);
        report_error("%s\n", __FUNCTION__);
    }

    int* new_sol;
    int i, n;
    fscanf(in, "%d", &n);
    if (n < 0 || n > INT_MAX)
        report_error("%s invalid solution file\n", __FUNCTION__);
    new_sol = new int[n + 2];
    new_sol[0] = n;
    for (i = 1; i <= n + 1; i++)
        fscanf(in, "%d", new_sol + i);

    // Import the buffer
    this->import_solution_buff(new_sol);
    fclose(in);
    delete[] new_sol;

    this->verify_routes("After read_solution_file\n");

    memcpy(this->best_sol_buff, this->current_sol_buff, (this->num_nodes + 2) * sizeof(int));

    return;

}
void VRP::add_time_constraints(std::vector<std::vector<double>> time_windows = { {-1} }, std::vector<double> service_times = { -1 }) {

    int max_id = 0;

#if TSPLIB_DEBUG
    printf("Loading time windows\n");
#endif

    // TIME_WINDOW_SECTION
    int i = 0;
    while (i <= num_nodes + 1)
    {
        nodes[i].start_tw = time_windows[i][0];
        nodes[i].end_tw = time_windows[i][1];
        i++;
    }


    // Fixed SERVICE_TIME
    if (service_times.size() == 1) {
        fixed_service_time = service_times[0];


#if TSPLIB_DEBUG
        printf("Setting service time to %f for all nodes\n", fixed_service_time);
#endif

        total_service_time = 0;
        for (i = 1; i <= num_nodes; i++)
        {
            // Set the service time to s for each non-depot node
            nodes[i].service_time = fixed_service_time;//+lcgrand(0) - use to test service times!
            total_service_time += nodes[i].service_time;
        }
        nodes[VRPH_DEPOT].service_time = 0;
        nodes[num_nodes + 1].service_time = 0;
        has_service_times = true;
    }


    // Variable service time
    has_service_times = true;

    i = 0;

    while (i <= num_nodes + 1)
    {
        nodes[i].service_time = service_times[i];
        total_service_time += nodes[i].service_time;
        if (nodes[i].id > max_id)
            max_id = nodes[i].id;
        i++;
    }
    nodes[num_nodes + 1].service_time = 0;

    //#if TSPLIB_DEBUG
    //          printf("Case 16\n");
    //#endif
    //
    //    // VEHICLES
    //    buff = strtok(NULL, "");
    //    myatoi(min_vehicles, buff);
    //    // This is not currently used
    //#if TSPLIB_DEBUG
    //          printf("Setting min_vehicles to %d\n", min_vehicles);
    //
    //#endif
}


void VRP::use_initial_solution(std::vector<std::vector<int>> sol) {

    int* new_sol;
    // n is the number of nodes
    int n = this-> num_nodes;

    //for (const auto& v : sol)
    //{
    //    for (int x : v)
    //    {
    //        n++;
    //    }
    //}

    std::vector<int>initial_sol(n + 1);

    int current_write = 0;

    for (int r = 0; r < sol.size(); r++) {
        initial_sol[current_write] = -sol[r][0];
        current_write++;
        for (int current_read = 1; current_read < sol[r].size(); current_read++) {
            initial_sol[current_write] = sol[r][current_read];
            current_write++;
        }
        //initial_sol[n] = 0;
    }
    

    //test vector pass through from python
    /*for (int i = 0; i < sol[0].size(); ++i) {
        std::cout << sol[0][i] << ' ';
    }
    printf("\n");*/

    //so far so good


    // make new_sol a list of size number nodes + 2
    new_sol = new int[n + 2];
    // the first element is the number of nodes
    new_sol[0] = n;

    // fill in the sol list with each element
    for (int i = 0; i <= n; i++) {
        new_sol[i + 1] = initial_sol[i];
        //printf("%d", initial_sol[i]);
    }
    //printf("\n");

    for (int j = 0; j < n+1; j++) {
        printf("% d ", new_sol[j]);
     }
    //printf("\n");

    // Import the buffer
    this->import_solution(new_sol);
    delete[] new_sol;
    initial_sol.clear();

    this->verify_routes("");

    memcpy(this->best_sol_buff, this->current_sol_buff, (this->num_nodes + 2) * sizeof(int));

    this->previous_total_route_length = this->get_total_route_length();
    this->export_solution_buff(this->previous_sol_buff);

    return;
}

void VRP::load_problem(
    int type, 
    std::vector<std::vector<double>> coordinates, 
    std::vector<double> demands, 
    std::vector<double> details, 
    std::vector<std::vector<double>> time_windows = { {-1} },
    int edge_type = -1, 
    int edge_format = -1
    ) {
    

    //////////////////////////////////////////////////////
    /*
    printf(" \n---------------\nloading input:\n");
    std::vector<double> v;
    std::cout << "\ntype: " << type << std::endl;
    std::cout << "\ncoords: [" << std::endl;
    for (int i = 0; i < int(coordinates.size()); i++){
        v = coordinates[i];
        for (int j = 0; j < int(v.size()); j++){
            std::cout << v[j] << ", ";
        }
        std::cout << "]" << std::endl << "[";
    }
    std::cout << "\ndemands: " << std::endl;
    for (int i = 0; i < int(demands.size()); i++){
        std::cout << demands[i] << ", ";
    }
    std::cout << "\ndetails: " << std::endl;
    for (int i = 0; i < int(details.size()); i++){
        std::cout << details[i] << ", ";
    }
    */
    //////////////////////////////////////////////////////


    int max_id = -VRP_INFINITY;
    int i;

    //bool has_depot = false;
    bool has_nodes = false;
    bool normalize = false;

    //check for time windows 
    bool has_time_windows = (time_windows[0][0] > -1);

    this->edge_weight_format = -1;
    this->edge_weight_type = -1;


    //set the problem type based on the arguments
    if (type == 0){
        problem_type = VRPH_TSP;
    }
    else if (type == 1) {
        problem_type = VRPH_CVRP;
    } 
    // not too sre if we need another problem type yet so for now we treat it as an addition
    /*else if (type == 3) {  
        problem_type = VRPH_CVRP;
    } */
    else {
        fprintf(stderr, "Unknown type encountered\n");
        report_error("%s\n", __FUNCTION__);
    }

#if TSPLIB_DEBUG
    printf("Problem type set\n");
#endif

    // set best known answer if available
    if (details[0] != -1)
        this->best_known = details[0];
    
    /*
    // set dimension 
    num_nodes = static_cast<int>(details[1]);
    // num_nodes is the # of non-VRPH_DEPOT nodes
    // The input N includes the VRPH_DEPOT for the VRP
    num_nodes--;
    // check input consistency
    if (num_nodes != num_original_nodes) {
        fprintf(stderr, "num_nodes provided in constructor (/w depot) and provided here are not consistent\n");
        report_error("%s\n", __FUNCTION__);
    }
    */

    matrix_size = num_nodes;
    dummy_index = num_nodes + 1;

    // CAPACITY
    if (details[1] > 0) {
        max_veh_capacity = static_cast<double>(details[1]);
        orig_max_veh_capacity = max_veh_capacity;
    }

    // DISTANCE
    if (details[2] > 0) {
        max_route_length = details[2];
        orig_max_route_length = max_route_length;
    }

    // NORMALIZATION
    if (details[3] > 0) {
        normalize = true;
    }

    // EDGE_WEIGHT_FORMAT - sets edge_weight_format
    if (edge_format > 0 && edge_format < 7)
        edge_weight_format = edge_format;
    else if (edge_format == -1)
        edge_weight_format = -1;
    else
    {
        // We didn't find a known format
        fprintf(stderr, "Unknown/Unsupported EDGE_WEIGHT_FORMAT %i encountered\n", edge_format);
        report_error("%s\n", __FUNCTION__);
    }

    // EDGE_WEIGHT_TYPE
    if (edge_type > -1 && edge_type < 10)
        edge_weight_type = edge_type;
    else if (edge_type == -1)
        edge_weight_type = -1;
    else
    {
        // We didn't find a known type
        fprintf(stderr, "Unknown/Unsupported EDGE_WEIGHT_TYPE %i encountered\n", edge_type);
        report_error("%s\n", __FUNCTION__);
    }

    // NODE_COORD_SECTION
    // Import the data 
    this->can_display = true;

    //load nodes
#if TSPLIB_DEBUG
    printf("num_nodes is %d\n", num_nodes);
#endif

    /////////////////////////////    
    //printf("\nload nodes...");
    ///////////////////////////// 


    i = 0;
    // coordinates[0][0] is VRP depot or TSP start node

    while (i <= num_nodes)  // loads all num_nodes+1 coords of depot and customer nodes
    {
        nodes[i].id = i+1;
        nodes[i].x = coordinates[i][0];
        nodes[i].y = coordinates[i][1];

        i++;

    }
    has_nodes = true;


    /*
    std::cout << "\nnodes: " << std::endl;
    for (int i = 0; i <= num_nodes+1; i++){
        std::cout << "(" << nodes[i].x << ", " << nodes[i].y << ") \n";
    }
    */


    // set additional dummy node as depot or starting node
    nodes[num_nodes + 1].x = nodes[0].x;
    nodes[num_nodes + 1].y = nodes[0].y;
    nodes[num_nodes + 1].id = 0;

    /*
    std::cout << "\nnodes: " << std::endl;
    for (int i = 0; i <= num_nodes+1; i++){
        std::cout << "(" << nodes[i].x << ", " << nodes[i].y << ") \n";
    }
    */
    if(has_service_times) {
        fprintf(stderr, "custom solver not implemented for <service_times> yet");
        report_error("%s\n", __FUNCTION__);
    }

    // patch_c
    // set the time windows if they are present
    if (has_time_windows) {
        i = 0;
        while (i <= num_nodes + 1)
        {
            nodes[i].start_tw = time_windows[i][0];
            nodes[i].end_tw = time_windows[i][1];
            i++;
        }
    }


    // set demands if available
    if (type == 1) {

        if(demands[0] != 0) {
            fprintf(stderr, "depot must have 0 demand but got %f\n", demands[0]);
            report_error("%s\n", __FUNCTION__);
        }

        i = 0;
        while (i <= num_nodes + 1)
        {

            //nodes[i].id = i;
            nodes[i].demand = demands[i];

            if (nodes[i].id > max_id)
                max_id = nodes[i].id;

            if (has_service_times == false)
                nodes[i].service_time = 0;
            i++;
        }
        nodes[num_nodes + 1].demand = 0;
        if(has_service_times==false)
            nodes[num_nodes+1].service_time=0;
    }


#if TSPLIB_DEBUG
    fprintf(stderr, "Loaded model, completing calculations...\n");
#endif
    // Now calculate distance matrix
    if (edge_weight_format == VRPH_FUNCTION || edge_weight_type != VRPH_EXPLICIT)
    {

#if TSPLIB_DEBUG
        printf("Creating distance matrix using edge_weight_type %d\n", edge_weight_type);
#endif

        //std::cout << "\nn_org=" << this->num_original_nodes << std::endl;
        //std::cout << "\nn=" << this->num_nodes << std::endl;

        //Allocate memory for the distance matrix
        if (d == NULL)
        {
            int n = num_nodes;
            int m = n + 2;

            d = new double* [m];
            d[0] = new double[(m) * (m)];
            for (i = 1; i < m; i++)
                d[i] = d[i - 1] + (m);
        }

        // Create the distance matrix using the appropriate 
        // distance function...
        create_distance_matrix();
    }


    // if specified in details, normalize coords to put the VRPH_DEPOT at the origin
    // if it is a standard EXACT_2D problem or EUC_2D problem
    if ((this->edge_weight_type == VRPH_EXACT_2D || this->edge_weight_type == VRPH_EUC_2D) && has_nodes && normalize && true)
    {
        this->max_theta = -VRP_INFINITY;
        this->min_theta = VRP_INFINITY;

        this->depot_normalized = true;
        double depot_x = nodes[0].x;
        double depot_y = nodes[0].y;

#if TSPLIB_DEBUG
        fprintf(stderr, "Normalizing...(%f,%f)\n", depot_x, depot_y);
#endif

        for (i = 0; i <= this->num_nodes + 1; i++)
        {
            nodes[i].x -= depot_x;
            nodes[i].y -= depot_y;
            // Translate everyone

            // Calculate the polar coordinates as well
            if (nodes[i].x == 0 && nodes[i].y == 0)
            {
                nodes[i].r = 0;
                nodes[i].theta = 0;
            }
            else
            {

                nodes[i].r = sqrt(((nodes[i].x) * (nodes[i].x)) + ((nodes[i].y) * (nodes[i].y)));
                nodes[i].theta = atan2(nodes[i].y, nodes[i].x);
                // We want theta in [0 , 2pi]
                // If y<0, add 2pi
                if (nodes[i].y < 0)
                    nodes[i].theta += 2 * VRPH_PI;
            }

            // Update min/max theta across all nodes - don't include the VRPH_DEPOT/dummy
            if (i != 0 && i != (this->num_nodes + 1))
            {
                if (nodes[i].theta > this->max_theta)
                    max_theta = nodes[i].theta;
                if (nodes[i].theta < this->min_theta)
                    min_theta = nodes[i].theta;
            }

        }
    }

#if TSPLIB_DEBUG
    fprintf(stderr, "Creating neighbor lists...\n");
#endif

    // Create the neighbor_lists-we may use a smaller size depending on the parameter
    // but we will construct the largest possible here...
    if (details[4]>1)
        this->create_neighbor_lists(VRPH_MIN(int(details[4]), num_nodes));
    else    
        this->create_neighbor_lists(VRPH_MIN(MAX_NEIGHBORLIST_SIZE, num_nodes));

}

// ==================================================================================================== //


void VRP::read_TSPLIB_file(const char *node_file)
{
    ///
    /// Processes each section of the provided TSPLIB file
    /// and records the relevant data in the VRP structure.  See the example
    /// files for information on my interpretation of the TSPLIB standard
    /// as it applies to VRP's.
    ///


   char *temp,*buff;
    char line[VRPH_STRING_SIZE];
    char *temp2;
    int max_id = -VRP_INFINITY;
	
    int ans;
    int x,y,i,j;
    float a, b;
    double s;

    bool has_depot=false;
    bool has_nodes=false;

    FILE *infile;

    infile  = fopen(node_file, "r");

    if (infile==NULL)
        report_error("%s: file error\n",__FUNCTION__);


    this->edge_weight_format=-1;
    this->edge_weight_type=-1;

    for(;;)
    {
        fgets(line, VRPH_STRING_SIZE-1, infile);

#if TSPLIB_DEBUG
        printf("Next line is %s\n",line);
#endif

        temp=strtok(line,":");

        // trim it (remove whitespace from the end and start)
        if(strlen(temp) > 0){
           while(*temp==' ') temp++;
           temp2 = temp+strlen(temp)-1;
           if (*temp2=='\n') temp2--;
           while(temp2>temp && *temp2==' ') temp2--;
           temp2[1] = '\0';
        }

#if TSPLIB_DEBUG
        printf("line begins with \"%s\"\n",temp);
#endif

        if( (ans=VRPCheckTSPLIBString(temp))<=0 )
        {
            if(ans==0)
            {
                fprintf(stderr,"Unknown string %s found\n",temp);
                report_error("%s\n",__FUNCTION__);
            }
            else
            {
                fprintf(stderr,"WARNING: TSPLIB string %s not supported\n",temp);
            }
        }

#if TSPLIB_DEBUG
        printf("ans is %d\n",ans);
#endif

        // Otherwise, we know the string - process it accordingly
        switch(ans)
        {
        case 1:
            // NAME

            temp2=strtok(NULL," ");
            memset(name,'\0',sizeof(char)*VRPH_STRING_SIZE);
            strncpy(name,temp2,VRPH_STRING_SIZE);//TODO: safe?       
            // Trim the ANNOYING \n if necessary...
            for(i=0;i<(int)strlen(name);i++)
            {
                if(name[i]=='\n' || name[i]=='\r')
                {
                    name[i]=0x00;
                    break;
                }
            }
        


#if TSPLIB_DEBUG
            printf("Name is %s\n",name);
#endif
            break;

        case 2:
            // TYPE
            temp2=strtok(NULL," ");

#if TSPLIB_DEBUG
            printf("Problem type is\n%s\n",temp2);
#endif
            if( (strncmp(temp2,"TSP",3)!= 0) && (strncmp(temp2,"CVRP",4)!=0)
                && (strncmp(temp2,"DCVRP",5)!=0) )
            {
                fprintf(stderr,"Unknown type %s encountered\n",temp2);
                report_error("%s\n",__FUNCTION__);
            }

            if( strncmp(temp2,"TSP",3)== 0) 
                problem_type=VRPH_TSP;
            if(strncmp(temp2,"CVRP",4)==0 || (strncmp(temp2,"DCVRP",5)!=0) )
                problem_type=VRPH_CVRP;

            break;


        case 3:
            // BEST_KNOWN
            temp2=strtok(NULL,"");
            this->best_known=atof(temp2);
            break;
        case 4:
            // DIMENSION
           //buff = strtok(NULL,"");
           //myatoi(num_nodes, buff);
           ////printf("%d", num_nodes);
           //num_nodes--;
           // matrix_size= num_nodes;
           
            num_nodes=atoi(strtok(NULL,""));
            num_nodes--;
            matrix_size = num_nodes;

            // num_nodes is the # of non-VRPH_DEPOT nodes
            // The value of DIMENSION includes the VRPH_DEPOT!
            dummy_index = 1+num_nodes;

            break;

        case 5:
            // CAPACITY
            max_veh_capacity = atoi(strtok(NULL, ""));
            orig_max_veh_capacity=max_veh_capacity;

#if TSPLIB_DEBUG
            printf("veh capacity is %g\n",max_veh_capacity);
#endif
            break;
        case 6:
            // DISTANCE
            max_route_length=(double)atof(strtok(NULL,""));
            orig_max_route_length=max_route_length;

#if TSPLIB_DEBUG
            printf("max length is %f\n",max_route_length);
#endif
            break;
        case 7:
            // EDGE_WEIGHT_FORMAT - sets edge_weight_format
            temp2=strtok(NULL," ");
            edge_weight_format=-1;

// these are equivalent for symmetric matrices
            if(strncmp(temp2,"UPPER_ROW",9)==0 || strncmp(temp2,"LOWER_COL",9)==0)
            {
                edge_weight_format=VRPH_UPPER_ROW;
            }

            if(strncmp(temp2,"FULL_MATRIX",11)==0)
            {
                edge_weight_format=VRPH_FULL_MATRIX;
            }
            
            if(strncmp(temp2,"FUNCTION",8)==0)
            {
                edge_weight_format=VRPH_FUNCTION;

            }
            // these are equivalent for symmetric matrices
            if(strncmp(temp2,"LOWER_ROW",9)==0 || strncmp(temp2,"UPPER_COL",9)==0)
            {
                edge_weight_format=VRPH_LOWER_ROW;

            }
            if(strncmp(temp2,"UPPER_DIAG_ROW",14)==0)
            {
                edge_weight_format=VRPH_UPPER_DIAG_ROW;

            }
            if(strncmp(temp2,"LOWER_DIAG_ROW",14)==0)
            {
                edge_weight_format=VRPH_LOWER_DIAG_ROW;

            }

            if(edge_weight_format == -1)
            {
                // We didn't find a known type
                fprintf(stderr,"Unknown/Unsupported EDGE_WEIGHT_FORMAT %s encountered\n",temp2);
                report_error("%s\n",__FUNCTION__);
            }
            break;
        case 8:  
            // EDGE_WEIGHT_TYPE
            edge_weight_type    = -1;
            temp2=strtok(NULL," ");


#if TSPLIB_DEBUG
            printf("Weight type is %s\n",temp2);
#endif
            
            // Determine the type of weight format
            if(strncmp(temp2,"EXPLICIT",8)==0)
            {
                edge_weight_type=VRPH_EXPLICIT;
            }

            if(strncmp(temp2,"EUC_2D",6)==0)
            {
                edge_weight_type=VRPH_EUC_2D;
            }

            if(strncmp(temp2,"EUC_3D",6)==0)
            {
                edge_weight_type=VRPH_EUC_3D;

            }

            if(strncmp(temp2,"MAX_2D",6)==0)
            {
                edge_weight_type=VRPH_MAX_2D;

            }

            if(strncmp(temp2,"MAX_3D",6)==0)
            {
                edge_weight_type=VRPH_MAX_3D;

            }

            if(strncmp(temp2,"MAN_2D",6)==0)
            {
                edge_weight_type=VRPH_MAN_2D;

            }

            if(strncmp(temp2,"MAN_3D",6)==0)
            {
                edge_weight_type=VRPH_MAN_3D;

            }

            if(strncmp(temp2,"CEIL_2D",7)==0)
            {
                edge_weight_type=VRPH_CEIL_2D;
            }

            if(strncmp(temp2,"GEO",3)==0)
            {
                edge_weight_type=VRPH_GEO;

            }

            if(strncmp(temp2,"EXACT_2D",8)==0)
            {
                edge_weight_type=VRPH_EXACT_2D;

            }

            if(edge_weight_type == -1)
            {
                // We didn't find a known type
                fprintf(stderr,"Unknown/Unsupported EDGE_WEIGHT_TYPE %s encountered\n",temp2);
                report_error("%s\n",__FUNCTION__);
            }        

            break;
        case 9:
            // NODE_COORD_TYPE - we don't really care about this one
            temp2=strtok(NULL," ");
            if( (strncmp(temp2,"TWOD_COORDS",11)!=0) && (strncmp(temp2,"THREED_COORDS",13)!=0) )
            {
                fprintf(stderr,"Unknown coordinate type %s encountered",temp2);
                report_error("%s\n",__FUNCTION__);
            }

            break;
        case 10:
            // EOF - clean up and exit
            fclose(infile);

#if TSPLIB_DEBUG
            fprintf(stderr,"Found EOF completing calculations...\n");
#endif

            this->max_theta= -VRP_INFINITY;
            this->min_theta=  VRP_INFINITY;

            // Now normalize everything to put the VRPH_DEPOT at the origin
            // if it is a standard EXACT_2D problem or EUC_2D problem


            // Now calculate distance matrix
            if (edge_weight_format == VRPH_FUNCTION || edge_weight_type != VRPH_EXPLICIT)
            {

#if TSPLIB_DEBUG
                printf("Creating distance matrix using edge_weight_type %d\n", edge_weight_type);
#endif

                //Allocate memory for the distance matrix
                if (d == NULL)
                {
                    int n = num_nodes;

                    d = new double* [n + 2];
                    d[0] = new double[(n + 2) * (n + 2)];
                    for (i = 1; i < n + 2; i++)
                        d[i] = d[i - 1] + (n + 2);
                }



                // Create the distance matrix using the appropriate 
                // distance function...
                create_distance_matrix();
            }

            if( (this->edge_weight_type==VRPH_EXACT_2D || this->edge_weight_type==VRPH_EUC_2D) && has_nodes && has_depot && true)
            {

                this->depot_normalized=true;
                double depot_x=nodes[0].x;
                double depot_y=nodes[0].y;

#if TSPLIB_DEBUG
                fprintf(stderr,"Normalizing...(%f,%f)\n",depot_x,depot_y);
#endif

                for(i=0;i<=this->num_nodes+1;i++)
                {
                    nodes[i].x -= depot_x;
                    nodes[i].y -= depot_y;
                    // Translate everyone

                    // Calculate the polar coordinates as well
                    if(nodes[i].x==0 && nodes[i].y==0)
                    {
                        nodes[i].r=0;
                        nodes[i].theta=0;
                    }
                    else
                    {

                        nodes[i].r=sqrt( ((nodes[i].x)*(nodes[i].x)) + ((nodes[i].y)*(nodes[i].y)) );
                        nodes[i].theta=atan2(nodes[i].y,nodes[i].x);
                        // We want theta in [0 , 2pi]
                        // If y<0, add 2pi
                        if(nodes[i].y<0)
                            nodes[i].theta+=2*VRPH_PI;
                    }

                    // Update min/max theta across all nodes - don't include the VRPH_DEPOT/dummy
                    if(i!=0 && i!=(this->num_nodes+1))
                    {
                        if(nodes[i].theta>this->max_theta)
                            max_theta=nodes[i].theta;
                        if(nodes[i].theta<this->min_theta)
                            min_theta=nodes[i].theta;
                    }

                }
            }

#if TSPLIB_DEBUG
                fprintf(stderr,"Creating neighbor lists...\n");
#endif

            // Create the neighbor_lists-we may use a smaller size depending on the parameter
            // but we will construct the largest possible here...
            this->create_neighbor_lists(VRPH_MIN(MAX_NEIGHBORLIST_SIZE,num_nodes));

#if TSPLIB_DEBUG
            fprintf(stderr,"Done w/ calculations...\n");
#endif
            return;

        case 11:
            // NODE_COORD_SECTION
            // Import the data 
            this->can_display=true;

#if TSPLIB_DEBUG
            printf("num_nodes is %d\n",num_nodes);
#endif

            i=0;
            x=0;
            while(i<=num_nodes)
            {
                fscanf(infile,"%d",&x);
                fscanf(infile,"%f",&a);
                fscanf(infile,"%f\n",&b);

                nodes[i].id= x;

                nodes[i].x=(double) a;
                nodes[i].y=(double) b;

                
                i++;

            }
            has_nodes=true;

            break;

        case 12:
            // DEPOT_SECTION
            // Load in the Depot Coordinates
            fscanf(infile,"%d\n",&x);
            if(x!=1)
            {
                fprintf(stderr,"Expected VRPH_DEPOT to be entry 1 - VRPH does not currently support"
                    " multiple DEPOTs\n");
                report_error("%s\n",__FUNCTION__);
            }
                    
#if TSPLIB_DEBUG
            //printf("VRPH_DEPOT has coordinates: %f, %f\n",a,b);
            printf("VRPH_DEPOT has coordinates: %f, %f\n",nodes[0].x,nodes[0].y);
#endif
            //nodes[0].x=a;
            //nodes[0].y=b;
            //nodes[0].id=0;

            has_depot=true;
            fscanf(infile,"%d\n",&x);
            if(x!= -1)
            {
                fprintf(stderr, "Expected -1 at end of DEPOT_SECTION.  Encountered %d instead\n",x);
                report_error("%s\n",__FUNCTION__);
            }

            // Now set the dummy node id to the VRPH_DEPOT!
            nodes[num_nodes+1].x=nodes[0].x;
            nodes[num_nodes+1].y=nodes[0].y;
            nodes[num_nodes+1].id=0;

            break;
        case 13:

#if TSPLIB_DEBUG
            printf("in case 13: DEMAND_SECTION w/ problem type %d # num_days=%d\n",problem_type,
                this->num_days);
#endif
            // DEMAND_SECTION
            // Read in the demands
                
            if(this->num_days<=1)
            {
                i=0;
                while(i<= num_nodes+1)
                {

                    fscanf(infile,"%d %d\n",&x,&y);
                    nodes[i].id=x;
                    nodes[i].demand=y;                

                    if(nodes[i].id>max_id)
                        max_id=nodes[i].id;

                    if(has_service_times==false)
                        nodes[i].service_time=0;
                    i++;
                }
                nodes[num_nodes+1].demand=0;
                if(has_service_times==false)
                    nodes[num_nodes+1].service_time=0;

            }
            else
            {
                // We have multiple days worth of demands
                i=0;
                while(i<=num_nodes+1)
                {
                    fscanf(infile,"%d",&x);
                    nodes[i].id=x;
                    for(j=1;j<this->num_days;j++)
                    {
                        fscanf(infile,"%d",&y);
                        this->nodes[i].daily_demands[j]=y;
                        
                    }
                    fscanf(infile,"%d\n",&y);
                    this->nodes[i].daily_demands[this->num_days]=y;
                    i++;
                    
                }
                this->nodes[num_nodes+1].demand=0;
                for(j=1;j<=this->num_days;j++)
                    this->nodes[num_nodes+1].daily_demands[j]=0;// Dummmy node
                
#if TSPLIB_DEBUG
                printf("Done with multiple day demands\n");
#endif
            }            


            break;

        case 14:

#if TSPLIB_DEBUG
            printf("Case 14\n");
#endif
           
            // EDGE_WEIGHT_SECTION

            // Make sure distance matrix is allocated
            if(d==NULL)
            {                
                d = new double* [num_nodes+2];
                d[0] = new double [(num_nodes+2)*(num_nodes+2)];
                for(i = 1; i < num_nodes+2; i++)    
                    d[i] = d[i-1] + (num_nodes+2) ;
            }

            if(edge_weight_format==VRPH_UPPER_DIAG_ROW)
            {
                // The zeros on the diagonal should be in the matrix!
                for(i=0;i<= num_nodes;i++)
                {
                    for(j=i;j<= num_nodes;j++)
                    {
                        fscanf(infile,"%lf",&(d[i][j]));
                        d[j][i]=d[i][j];
                    }
                    // Add in column for the dummy-assumed to live at the VRPH_DEPOT
                    d[i][num_nodes+1]=d[i][0];
                }
                // Now add a row for the dummy
                for(j=0;j<=num_nodes+1;j++)
                    d[num_nodes+1][j]=d[0][j];

            }                

            if(edge_weight_format==VRPH_FULL_MATRIX)
            {
                this->symmetric=false;

                for(i=0;i<= num_nodes;i++)
                {
                    for(j=0;j<= num_nodes;j++)
                    {
                        fscanf(infile,"%lf",&(d[i][j]));
                    }
                    // Add in column for the dummy-assumed to live at the VRPH_DEPOT
                    d[i][num_nodes+1]=d[i][0];

                }
                // Now add a row for the dummy
                for(j=0;j<=num_nodes+1;j++)
                    d[num_nodes+1][j]=d[0][j];

            }
            
            // LOWER_DIAG_ROW format
            if(edge_weight_format==VRPH_LOWER_DIAG_ROW)
            {
                // The zeros on the diagonal should be in the matrix!
                for(i=0;i<= num_nodes;i++)
                {
                    for(j=0;j<= i;j++)
                    {
                        fscanf(infile,"%lf",&(d[i][j]));
                        d[j][i]=d[i][j];
                    }
                    // Add in column for the dummy-assumed to live at the VRPH_DEPOT
                    d[i][num_nodes+1]=d[i][0];

                }
                // Now add a row for the dummy
                for(j=0;j<=num_nodes+1;j++)
                    d[num_nodes+1][j]=d[0][j];

            }

            // UPPER_ROW format - no diagonal
            if(edge_weight_format==VRPH_UPPER_ROW)
            {
                for(i=0;i<=num_nodes;i++)
                {
                    for(j=i+1;j<= num_nodes;j++)
                    {
                        fscanf(infile,"%lf",&(d[i][j]));
                        d[j][i]=d[i][j];
                    }
                    // Add in column for the dummy-assumed to live at the VRPH_DEPOT
                    d[i][num_nodes+1]=d[i][0];
                    d[i][i]=0;

                }
                // Now add a row for the dummy
                for(j=0;j<=num_nodes+1;j++)
                    d[num_nodes+1][j]=d[0][j];

            }

             // LOWER_ROW format
            if(edge_weight_format==VRPH_LOWER_ROW)
            {
                for(i=0;i<= num_nodes;i++)
                {
                    for(j=0;j<i;j++)
                    {
                        fscanf(infile,"%lf",&(d[i][j]));
                        d[j][i]=d[i][j];
                    }
                    d[i][i]=0;
                    // Add in column for the dummy-assumed to live at the VRPH_DEPOT
                    d[i][num_nodes+1]=d[i][0];

                }
                // Now add a row for the dummy
                for(j=0;j<=num_nodes+1;j++)
                    d[num_nodes+1][j]=d[0][j];
            }

            // The last fscanf doesn't get the newline...
            fscanf(infile,"\n");
            break;

        case 15:

#if TSPLIB_DEBUG
            printf("Case 15\n");
#endif

            // SERVICE_TIME

            s=(double)(atof(strtok(NULL,"")));
            fixed_service_time=s;
#if TSPLIB_DEBUG
            printf("Setting service time to %f for all nodes\n", fixed_service_time);
#endif
            total_service_time=0;
            for(i=1;i<=num_nodes;i++)
            {
                // Set the service time to s for each non-depot node
                nodes[i].service_time=s;//+lcgrand(0) - use to test service times!
                total_service_time+=nodes[i].service_time;
            }
            nodes[VRPH_DEPOT].service_time=0;
            nodes[num_nodes+1].service_time=0;

            has_service_times=true;

            break;

        case 16:

#if TSPLIB_DEBUG
            printf("Case 16\n");
#endif

            // VEHICLES
            buff = strtok(NULL,"");
            myatoi(min_vehicles,buff);
            // This is not currently used
#if TSPLIB_DEBUG
            printf("Setting min_vehicles to %d\n",min_vehicles);

#endif
            break;

        case 17:
        
            // NUM_DAYS
           //this->num_days=atoi(strtok(NULL,""));
           buff = strtok(NULL,"");
           myatoi(this->num_days,buff);
           break;

        case 18:
            // SVC_TIME_SECTION
            
            has_service_times=true;

            if(this->num_days==0)// Standard problem
            {
                i=0;
                while(i<= num_nodes)
                {

                    fscanf(infile,"%d %lf\n",&x,&s);
                    nodes[i].id=x;
                    nodes[i].service_time=s;                
                    total_service_time+=nodes[i].service_time;

                    if(nodes[i].id>max_id)
                        max_id=nodes[i].id;
                    i++;
                }
                nodes[num_nodes+1].service_time=0;

                
            }
            else
            {
                // We have multiple days worth of service times
                i=0;
                while(i<=num_nodes)
                {
                    fscanf(infile,"%d",&x);
                    nodes[i].id=x;
                    for(j=1;j<this->num_days;j++)
                    {
                        fscanf(infile,"%lf",&s);
                        this->nodes[i].daily_service_times[j]=s;
                    }
                    fscanf(infile,"%lf\n",&s);
                    this->nodes[i].daily_service_times[this->num_days]=s;
                    i++;
                    
                }
                this->nodes[num_nodes+1].service_time=0;
                for(j=1;j<=this->num_days;j++)
                    this->nodes[num_nodes+1].daily_service_times[j]=0;// Dummmy node
                
            }        

            
            break;
        case 19:
            // TIME_WINDOW_SECTION
            i=0;
            while(i<= num_nodes+1)
            {
                fscanf(infile,"%d %f %f\n",&x,&a,&b);
                nodes[i].start_tw=a;
                nodes[i].end_tw=b;
                i++;
            }

            break;
        case 20:
            // COMMENT
            // Don't care
            break;
        case 21:
            // DISPLAY_DATA_SECTION
            // Overwrite node's x and y coords with this data

            this->can_display=true;
            i=0;
            x=0;
            while(i<=num_nodes)
            {
                fscanf(infile,"%d",&x);
                if(x < 0 || x > INT_MAX)
                   report_error("%s display data invalid\n",__FUNCTION__);
                fscanf(infile,"%f",&a);
                fscanf(infile,"%f\n",&b);

                nodes[x-1].x=(double) a;
                nodes[x-1].y=(double) b;                
                i++;

            }

            break;
        case 22:
            // TWOD_DISPLAY
            break;
        case 23:
            // DISPLAY_DATA_TYPE

            break;
        case 24:
            // NO_DISPLAY
            break;
        case 25:
            // COORD_DISPLAY
            break;
        }
    }

}    

void VRP::write_TSPLIB_stream(FILE* out)
{
    int i,j;
        
    fprintf(out,"NAME: %s\n",name);
    fprintf(out,"TYPE: CVRP\n");
    if(this->best_known!=-1)
        fprintf(out,"BEST_KNOWN: %5.3f\n", this->best_known);
    fprintf(out,"DIMENSION: %d\n",num_nodes+1);
    fprintf(out,"CAPACITY: %g\n",max_veh_capacity);
    if(max_route_length!=VRP_INFINITY)
        fprintf(out,"DISTANCE: %4.5f\n",max_route_length);
    if(min_vehicles!=-1)
        fprintf(out,"VEHICLES: %d\n",min_vehicles);
    fprintf(out,"EDGE_WEIGHT_TYPE: EXPLICIT\n");
    fprintf(out,"EDGE_WEIGHT_FORMAT: LOWER_ROW\n");
    fprintf(out,"NODE_COORD_TYPE: TWOD_COORDS\n");
    fprintf(out,"NODE_COORD_SECTION\n");

    // Start numbering at 1!!
    fprintf(out,"%d %4.5f %4.5f\n",1,nodes[0].x,nodes[0].y);
    for(i=1;i<=num_nodes;i++)
    {
        fprintf(out,"%d %4.5f %4.5f\n",i+1,nodes[i].x,nodes[i].y);
    }
    
    fprintf(out,"EDGE_WEIGHT_SECTION\n");
    for(i=1;i<=num_nodes;i++)
    {
        for(j=0;j<i;j++)
        {
            fprintf(out,"%4.2f ", (this->d[i][j]));
        }
        fprintf(out,"\n");
    }

    fprintf(out,"DEMAND_SECTION\n");
    // Start numbering at 1!!
    fprintf(out,"1 0\n");
    for(i=1;i<=num_nodes;i++)
    {
        fprintf(out,"%d %g\n",i+1,nodes[i].demand);
    }

    fprintf(out,"DEPOT_SECTION\n");
    fprintf(out,"1\n-1\n");
}

void VRP::write_TSPLIB_file(const char *outfile)
{
    /// 
    /// Exports the data from an already loaded instance
    /// to a CVRP outfile in TSPLIB format (using EXPLICIT / LOWER_ROW distances
    /// to make sure the problem is in correct format for all SYMMETRIC problems ).
    ///

    FILE *out;
    if( (out=fopen(outfile,"w"))==NULL)
    {
        report_error("%s: Can't open file for writing...\n",__FUNCTION__);
    }
    else
    {
        write_TSPLIB_stream(out);
        fprintf(out,"EOF\n");
        fclose(out);
    }
}

void VRP::write_solution_file(const char *filename)
{
    ///
    /// Exports a solution to the designated filename in canonical form.  
    /// Let N be the # of non-VRPH_DEPOT nodes in the problem. Then the first entry in the file
    /// is N and the following N+1 entries simply traverse the solution in order where we enter
    /// a node's negative index if it is the first node in a route.
    /// The solution is put into canonical form - the routes are traversed in the orientation
    /// where the start index is less than the end index, and the routes are sorted by
    /// the start index.
    /// Example:    Route 1:  0-3-2-0, Route 2:  0-4-1-0
    /// File is then:
    /// 4 -1 4 -2 3 0    
    ///



    int n, current;
    FILE *out;

    int *sol;

    // Open the file
    if( (out = fopen(filename,"w"))==NULL)
    {
        fprintf(stderr,"Error opening %s for writing\n",filename);
        report_error("%s\n",__FUNCTION__);
    }

    // First, determine the # of nodes in the Solution -- this could be different than the
    // VRP.num_nodes value if we are solving a subproblem

    n=0;
    current=VRPH_ABS(next_array[VRPH_DEPOT]);
    while(current!=VRPH_DEPOT)
    {
        current=VRPH_ABS(next_array[current]);
        n++;
    }
    // We have n non-VRPH_DEPOT nodes in the problem
    sol=new int[n+2];
    // Canonicalize
    this->export_canonical_solution_buff(sol);
    this->import_solution_buff(sol);
    fprintf(out,"%d ",n);


    // Now output the ordering - this is actually just the sol buffer
    current=next_array[VRPH_DEPOT];
    fprintf(out,"%d ",current);
    while(current!=VRPH_DEPOT)
    {
        current=next_array[VRPH_ABS(current)];
        fprintf(out,"%d ",current);

    }

    fprintf(out,"\n\n\n");


    fflush(out);
    fprintf(out,"\n\n\nOBJ=\n%5.3f\nBEST_KNOWN=\n%5.3f",this->total_route_length-this->total_service_time,
        this->best_known);

    fflush(out);
    fclose(out);

    delete [] sol;
    return;
}

void VRP::write_solutions(int num_sols, const char *filename)
{
    ///
    /// Writes num_sols solutions from the solution warehouse to the designated filename. 
    /// The format is the same as for write_solution_file.
    ///

    int i,n, current;
    FILE *out;
    int *sol;

    sol=new int[this->num_original_nodes+2]; // should be big enough

    // Open the file
    if( (out = fopen(filename,"w"))==NULL)
    {
        fprintf(stderr,"Error opening %s for writing\n",filename);
        report_error("%s\n",__FUNCTION__);
    }

    for(i=0;i<num_sols;i++)
    {
        if(i>this->solution_wh->num_sols)
            report_error("%s: too many solutions!\n",__FUNCTION__);

        // Import this solution and then write it out
        this->import_solution_buff(this->solution_wh->sols[i].sol);
        this->export_canonical_solution_buff(sol);
        this->import_solution_buff(sol);


        // First, determine the # of nodes in the Solution -- this could be different than the
        // VRP.num_nodes value if we are solving a subproblem

        n=0;
        current=VRPH_ABS(next_array[VRPH_DEPOT]);
        while(current!=VRPH_DEPOT)
        {
            current=VRPH_ABS(next_array[current]);
            n++;
        }
        // We have n non-VRPH_DEPOT nodes in the problem
        fprintf(out,"%d ",n);

        // Now output the ordering
        current=next_array[VRPH_DEPOT];
        fprintf(out,"%d ",current);
        while(current!=VRPH_DEPOT)
        {
            current=next_array[VRPH_ABS(current)];
            fprintf(out,"%d ",current);

        }
        fprintf(out,"\n");
    }

    fflush(out);
    fclose(out);
    delete [] sol;
    return;

}

void VRP::write_tex_file(const char *filename)
{
    ///
    /// Writes the solution in a TeX tabular format using the
    /// longtable package in case the solution spans multiple
    /// pages.
    ///

    int i;

    FILE *out;
    if( (out=fopen(filename,"w"))==NULL)
    {
        fprintf(stderr,"Error opening %s for writing\n",filename);
        report_error("%s\n",__FUNCTION__);
        
    }    

    // The headers for the first page and pages thereafter
    fprintf(out,"%% TeX file automatically generated by VRPH for problem %s\n\n",this->name);
    fprintf(out,"\\renewcommand{\\baselinestretch}{1}\n");
    fprintf(out,"\\footnotesize\n");
    fprintf(out,"\\begin{center}\n");
    fprintf(out,"\\begin{longtable}{|c|r|r|p{4 in}|}\n");

    fprintf(out,"\\hline\n");
    fprintf(out,"Route&\\multicolumn{1}{c|}{Length}&\\multicolumn{1}{c|}{Load}&\\multicolumn{1}{c|}{Ordering}\\\\\n");
    fprintf(out,"\\hline\n");
    fprintf(out,"\\endhead\n");
    fprintf(out,"\\hline\n");
    fprintf(out,"\\multicolumn{3}{|l|}{Problem}&%s\\\\\n",this->name);
    fprintf(out,"\\hline\n");
    fprintf(out,"\\endfirsthead\n");
    fprintf(out,"\\endfoot\n");
    fprintf(out,"\\endlastfoot\n");
    fprintf(out,"\\multicolumn{3}{|l|}{Vehicle capacity}&%g\\\\\n",this->max_veh_capacity);
    if(this->max_route_length!=VRP_INFINITY)
        fprintf(out,"\\multicolumn{3}{|l|}{Maximum route length}&%5.3f\\\\\n",this->max_route_length);
    else
        fprintf(out,"\\multicolumn{3}{|l|}{Maximum route length}&N/A\\\\\n");
    if(this->total_service_time>0)
        fprintf(out,"\\multicolumn{3}{|l|}{Total service time}&%5.3f\\\\\n",this->total_service_time);
    fprintf(out,"\\multicolumn{3}{|l|}{Number of nodes}&%d\\\\\n",this->num_nodes);
    fprintf(out,"\\multicolumn{3}{|l|}{Total route length}&%5.3f\\\\\n",this->total_route_length-
        this->total_service_time);
    fprintf(out,"\\multicolumn{3}{|l|}{Total number of routes}&%d\\\\\n",this->total_number_of_routes);
    fprintf(out,"\\hline\n");
    fprintf(out,"Route&\\multicolumn{1}{c|}{Length}&\\multicolumn{1}{c|}{Load}&\\multicolumn{1}{c|}{Ordering}\\\\\n");
    fprintf(out,"\\hline\n");

    for(i=1;i<=this->total_number_of_routes;i++)
    {
        fprintf(out,"%d&%5.3f&%g&(0",i,this->route[i].length,this->route[i].load);
        int current=this->route[i].start;
        while(current>=0)
        {
            fprintf(out,", %d",current);
            current=this->next_array[current];
            
        }
        fprintf(out,", 0)\\\\\n");
        fprintf(out,"\\hline\n");
    }
    fprintf(out,"\\caption{The best solution found for problem %s}\n",this->name);
    fprintf(out,"\\end{longtable}\n");
    fprintf(out,"\\end{center}\n");
    fprintf(out,"\\renewcommand{\\baselinestretch}{2}\n");
    fprintf(out,"\\normalsize\n");
    fclose(out);
}

void VRP::read_optimum_file(const char *filename)
{
    ///
    /// Imports an optimal solution from filename. File is in TSPLIB
    ///  format.
    ///

    FILE *in;

    if( (in=fopen(filename,"r"))==NULL)
    {
        fprintf(stderr,"Error opening %s for reading\n",filename);
        report_error("%s\n",__FUNCTION__);
    }

    int *new_sol;
    //int current_route = 0;
    int j;
    int node;
    char* next_char;
    
    // assume that we have already loaded the vrp problem and 
    //  we know the number of nodes
    new_sol = new int[num_nodes + 2];
    next_char = new char[2];
    new_sol[0]=num_nodes;

    // the rows are of format
    // Route #2: 21 17 12 
    j = 1;
    while (j<=num_nodes)
    {
        if (fgets(next_char, 2, in) == 0)
            break;
        if (feof(in) != 0)
            break;
        if (next_char[0] == ':')
        {
            fscanf(in, " %d", &node);
            new_sol[j++] = -node;
            while (fscanf(in, " %d", &node)==1)
            {
                new_sol[j++] = node;
            }
        }
    }
    while (j <= num_nodes + 1)
    {
        new_sol[j++] = 0;
    }
    fclose(in);
    
    // Import the buffer
    this->import_solution_buff(new_sol);
    
    // clean up
    delete [] new_sol;
    delete[] next_char;

    this->verify_routes("After read_solution_file\n");

    memcpy(this->best_sol_buff,this->current_sol_buff,(this->num_nodes+2)*sizeof(int));

    return;
}

void VRP::import_solution_buff(int *sol_buff)
{
    ///
    /// Imports a solution from buffer produced by something like
    /// export_solution_buff().  Can be a partial solution if the first
    /// element in sol_buff[] is less than num_original_nodes;
    ///
    /// JHR: Just a comment. This mehtod is a complete mess. I suggest a complete rewrite.
    ///  It would also make it easier to undestrand if some kind of notation would
    ///  be used to member variables.
    ///


    int i, n, rnum, current, next, num_in_route;
    double len, load;

    next=-1; //to avoid warning...

    // Set all nodes to unrouted
    for(i=1;i<=this->num_original_nodes;i++)
        routed[i]=false;

    len=0;
    load=0;
    rnum=0;

    this->total_route_length = 0;
    this->num_nodes = sol_buff[0];
    this->total_number_of_routes = 0;
    
    n=this->num_nodes;

    // JHR: If importing a solution with no routes
    //  Verify breaks if route 0 is not verified. I just hope that initializing
    //  the route[0] as I did it breaks nothing.
    if (n==0)
    {
        next_array[VRPH_DEPOT] = VRPH_DEPOT;
        route[0].start=VRPH_DEPOT;
        route[0].end=VRPH_DEPOT;
        route[0].length=0;
        route[0].load=0;
        route[0].num_customers=0;
    }
    else
    {
        // Route bookkeeping
        num_in_route=0;
        rnum=1;

        current=VRPH_ABS(sol_buff[1]);
        routed[VRPH_ABS(current)]=true;
        next_array[VRPH_DEPOT]=sol_buff[1];
        route_num[VRPH_ABS(current)]=rnum;
        route[rnum].start=VRPH_ABS(current);
        load+=nodes[VRPH_ABS(current)].demand;
        len+=d[VRPH_DEPOT][VRPH_ABS(current)];
        num_in_route++;

        // JHR: If there is only 1 
        if (n==1)
        {
            next = current;
        }
    
        for(i=2;i<=n;i++)
        {
            next=sol_buff[i];
            routed[VRPH_ABS(next)]=true;

            if(next<0)
            {
                // end of route - current is the last node in this route
                // next is the first node in the next route

                len+=d[VRPH_ABS(current)][VRPH_DEPOT];

                route[rnum].end=VRPH_ABS(current);
                route[rnum].length=len;
                route[rnum].load=load;
                route[rnum].num_customers=num_in_route;
                total_route_length+=len;

                if(rnum>n)
                {
                    fprintf(stderr,"%d>%d:  rnum too big in import solution buff!\n",rnum,n);
                    for(int j=0;j<=n;j++)
                        fprintf(stderr,"%d ",sol_buff[j]);
        
                    report_error("%s\n",__FUNCTION__);
                }

                
                rnum++;
                num_in_route=0;
                len=0;
                load=0;
                len+=d[VRPH_DEPOT][VRPH_ABS(next)];
                route_num[VRPH_ABS(next)]=rnum;
                route[rnum].start=VRPH_ABS(next);
            }
            else
                // Not at the end of a route
                len+=d[VRPH_ABS(current)][VRPH_ABS(next)];



            next_array[VRPH_ABS(current)]=next;
            current=next;

            load+=nodes[VRPH_ABS(current)].demand;
            num_in_route++;

            route_num[VRPH_ABS(current)]=rnum;
        }

        next_array[VRPH_ABS(next)]=VRPH_DEPOT;
        route_num[VRPH_ABS(next)]=rnum;

        len+=d[VRPH_ABS(next)][VRPH_DEPOT];

        route[rnum].end=VRPH_ABS(next);
        route[rnum].length=len;
        route[rnum].load=load;
        route[rnum].num_customers=num_in_route;
        total_route_length+=len;
    }
    total_number_of_routes=rnum;
    create_pred_array();

#if VERIFY_ALL
    // Make sure everything imported successfully!
    verify_routes("After import sol_buff");
#endif

    // Mark all the nodes as "routed"
    for(i=1;i<=sol_buff[0];i++)
        routed[VRPH_ABS(sol_buff[i])]=true;

    routed[VRPH_DEPOT]=true;

    route_num[VRPH_DEPOT]=0;

    memcpy(this->current_sol_buff,sol_buff,(this->num_nodes+2)*sizeof(int));

    return;

}

void VRP::import_solution(int * sol_buff)
{
   ///
   /// Imports a solution from buffer and marks it as best.
   ///
   this->import_solution_buff(sol_buff);
#if VERIFY_ALL
   assert(this->verify_routes("After import_solution\n"));
#endif
   memcpy(this->best_sol_buff, this->current_sol_buff, (this->num_nodes + 2) * sizeof(int));
   this->set_best_total_route_length(this->get_total_route_length());
   return;
}

void VRP::export_solution_buff(int *sol_buff)
{
    ///
    /// Exports the solution to sol_buff.
    ///

    int i, current;

    sol_buff[0]=num_nodes;    

    // Now output the ordering
    current=next_array[VRPH_DEPOT];
    sol_buff[1]=current;
    i=2;

    while(current!=VRPH_DEPOT)
    {
        current=next_array[VRPH_ABS(current)];
        sol_buff[i]=current;
        i++;
    }

    return;
}

void VRP::export_canonical_solution_buff(int *sol_buff)
{
    ///
    /// Puts the solution into the buffer in a "canonical form".
    /// The orientation of each route is such that start<end.
    /// Also, the ordering of the different routes is determined
    /// so that route i precedes route j in the ordering if
    /// start_i < start_j.
    ///

    int i,j,next;
    int *start_buff;

    //printf("VRP::export_canonical_solution_buff total_route_length:%g\n",total_route_length);
    start_buff=new int[total_number_of_routes];
    
    this->normalize_route_numbers();

    // First orient each route properly
    for(i=1;i<=total_number_of_routes;i++)
    {
       //We cannot reverse the route if we are asymmetric without messing up the length.
       if(this->symmetric && route[i].end<route[i].start){
          reverse_route(i);
          //printf("VRP::export_canonical_solution_buff after reverse total_route_length:%g\n",total_route_length);
       }
        start_buff[i-1]=route[i].start;
    }


    // Sort the start_buff
    qsort(start_buff, total_number_of_routes, sizeof(int), VRPIntCompare);

    sol_buff[0]=this->num_nodes;

    // Now order the routes themselves
    j=1;
    for(i=0;i<total_number_of_routes;i++)
    {
        sol_buff[j]=-start_buff[i];
        for(;;)
        {
            next=this->next_array[VRPH_ABS(sol_buff[j])];
            if(next<=0)
                break; // next route

            j++;
            sol_buff[j]=next;
        }
        j++;
    }
    
    sol_buff[j]=VRPH_DEPOT;

    delete [] start_buff;

    return;

}

void VRP::show_routes()
{
    ///
    /// Displays all routes in the solution.
    ///


    int i, cnt;
    int route_start;
    int next_node_number;
    int current_node, current_route;
    double total_load = 0;


    printf("-----------------------------------------------\n");
    printf("Total route length:  %5.2f\n",total_route_length);
    i = 1;
    cnt = 0;
    route_start = -next_array[VRPH_DEPOT];
    current_node = route_start;
    current_route = route_num[current_node];
    total_load+= route[current_route].load;


    printf("\nRoute %04d(routenum=%d)[0-%d...%d-0, %5.2f, %g, %d]: \n",i,current_route,
        nodes[route[current_route].start].id-1,
        nodes[route[current_route].end].id-1,
        route[current_route].length,
        route[current_route].load,
        route[current_route].num_customers);

    printf("%d-%d-",VRPH_DEPOT,nodes[route_start].id-1);

    cnt++;

    while(route_start != 0 && i<num_nodes+1)
    {
        // When we finally get a route beginning at 0, this is the last route 
        // and there is no next route, so break out
        if(next_array[current_node]==0)
        {
            printf("%d\n",VRPH_DEPOT);
            printf("End of routes.  Totals: (%d routes,%d nodes,%g total load)\n",i,cnt,total_load);
            printf("-----------------------------------------------\n");
            if(cnt!= num_nodes)
            {
                fprintf(stderr,"Not enough nodes! counted=%d; claimed=%d\n",cnt,num_nodes);
                report_error("%s\n",__FUNCTION__);
            }

            return;
        }

        if(next_array[current_node]>0)
        {
            // Next node is somewhere in the middle of a route
            next_node_number = next_array[current_node];
            printf("%d-",nodes[next_node_number].id-1);

            current_node=next_node_number;
            cnt++;

            if(cnt>num_nodes)
            {
                fprintf(stderr,"Too many nodes--cycle?\n");
                fprintf(stderr,"Count is %d, num_nodes is %d\n",cnt,num_nodes);
                show_next_array();
                report_error("%s\n",__FUNCTION__);
            }
        }
        else
        {
            // We must have a non-positive "next" node indicating the beginning of a new route
            i++;
            printf("%d",VRPH_DEPOT);

            route_start = - (next_array[current_node]);    
            current_route = route_num[route_start];
            current_node = route_start;

            printf("\n\nRoute %04d(routenum=%d)[0-%d...%d-0, %3.2f, %g, %d]: \n",i,current_route,
                nodes[route[current_route].start].id-1,
                nodes[route[current_route].end].id-1,
                route[current_route].length,
                route[current_route].load,
                route[current_route].num_customers);


            // Print out the beginning of this route
            total_load += route[current_route].load;
            printf("%d-%d-",VRPH_DEPOT,nodes[current_node].id-1);
            cnt++;
        }
    }

    
}

void VRP::show_route(int k)
{
    ///
    /// Displays information about route number k.
    ///

    int current_node;
    int i=0;
    if(k<=0)
        report_error("%s: called with non-positive route number\n",__FUNCTION__);

    printf("\nRoute %03d[0-%03d...%03d-0, %5.3f, %g, %d]: \n",k,
        route[k].start,
        route[k].end,
        route[k].length,
        route[k].load,
        route[k].num_customers);

    printf("%d-",VRPH_DEPOT);

    current_node= route[k].start;
    while(current_node != route[k].end)
    {
        printf("%03d-",current_node);
        current_node= next_array[current_node];
        i++;
        if(i>num_nodes)
            report_error("%s: encountered too many nodes!!\n",__FUNCTION__);
    }
    printf("%03d-%d\n\n",current_node,VRPH_DEPOT);

}

std::vector<int> VRP::get_route(int k)
{
    ///
    /// Return information about route number k.
    ///

    std::vector<int> route_cp;

    int current_node;
    int i = 0;

    if (k <= 0)
        report_error("%s: called with non-positive route number\n", __FUNCTION__);

    current_node = route[k].start;

    //iterate over every node in the route
    while (current_node != route[k].end)
    {
        route_cp.push_back(current_node);
        current_node = next_array[current_node];
        i++;
        if (i > num_nodes)
            report_error("%s: encountered too many nodes!!\n", __FUNCTION__);
    }
    //add the last node in the route
    route_cp.push_back(current_node);
    route_cp.push_back(0);
    return route_cp;
}

std::vector<std::vector<int>> VRP::get_routes()
{
    ///
    /// Returns a vector of all routes in the solution.
    ///

    std::vector<std::vector<int>> routes_cp;        //holds the final total routes
    std::vector<int> temp_route;                    //holds the temp current route

    int total_route_number = this->get_total_number_of_routes();
    int current_node;
    int i = 0;

    //iterate over the current routes
    for (int current_route = 1; current_route <= total_route_number; current_route++) {
        
        //start with the first node in the list
        current_node = route[current_route].start;

        //iterate over the total node in the current route
        while (current_node != route[current_route].end)
        {
            temp_route.push_back(current_node);
            current_node = next_array[current_node];
            i++;
            if (i > num_nodes)
                report_error("%s: encountered too many nodes!!\n", __FUNCTION__);
        }
        //add the last node
        temp_route.push_back(current_node);
        temp_route.push_back(0);
        //route is done, save it and repeat
        routes_cp.push_back(temp_route);
        temp_route.clear();
    }

    return routes_cp;

}


std::vector<std::vector<double>> VRP::get_dynamic_node_data(){

    std::vector<std::vector<double>> dyn_features;    //holds the dynamic node features
    std::vector<double> temp_buffer;

    // update features
    this->update_dynamic_node_data();

    // collect and return
    for (int current_node = 0; current_node <= num_nodes; current_node++) {
        temp_buffer.push_back(nodes[current_node].cummulative_distance);
        temp_buffer.push_back(nodes[current_node].cummulative_distance_back);
        temp_buffer.push_back(nodes[current_node].cummulative_demand);
        temp_buffer.push_back(nodes[current_node].cummulative_demand_back);
        dyn_features.push_back(temp_buffer);
        temp_buffer.clear();
    }

   return dyn_features;

}


void VRP::summary()
{
    ///
    /// This function prints out a summary of the current solution and the individual routes.
    ///

    int i, cnt;
    int route_start;
    int next_node_number;
    int current_node, current_route;
    double total_load = 0;
    int num_in_route=0;
    int total_nodes=0;
    int cust_count=0;
    bool feasible=true;

    printf("\n------------------------------------------------\n");
    printf("Solution for problem %s\n",this->name);
    printf("Total route length:       %5.2f\n",total_route_length-this->total_service_time);
    if(this->best_known!=VRP_INFINITY)
        printf("Best known solution:      %5.2f\n",this->best_known);
    printf("Total service time:       %5.2f\n",this->total_service_time);
    if(this->max_route_length!=VRP_INFINITY)
        printf("Vehicle max route length: %5.2f\n",this->max_route_length);
    else
        printf("Vehicle max route length: N/A\n");
    printf("Vehicle capacity:         %g\n",this->max_veh_capacity);
    printf("Number of nodes visited:  %d\n",this->num_nodes);
    printf("------------------------------------------------\n");
    i = 1;
    cnt = 0;
    route_start = -next_array[VRPH_DEPOT];
    current_node = route_start;
    current_route = route_num[current_node];
    total_load+= route[current_route].load;


    printf("\nRoute %03d[0-%03d...%03d-0\tlen=%03.2f\tload=%04g\t#=%03d]",i,route[current_route].start,
        route[current_route].end,route[current_route].length,
        route[current_route].load,route[current_route].num_customers);
    // Check feasibility
    if(route[current_route].length>this->max_route_length || 
        route[current_route].load > this->max_veh_capacity)
        feasible=false;
    cust_count+= route[current_route].num_customers;

    if(current_node!= dummy_index)
        num_in_route=1;

    total_nodes++;

    cnt++;

    while(route_start != 0 && i<num_nodes+1)
    {
        // When we finally get a route beginning at 0, this is the last route 
        // and there is no next route, so break out
        if(next_array[current_node]==0)
        {
            num_in_route=0;
            if(cnt!= num_nodes)
            {
                fprintf(stderr,"Not enough nodes:  counted=%d; claimed=%d!\n",cnt,num_nodes);
                report_error("%s\n",__FUNCTION__);
            }

            printf("\n\n");
            if(!feasible)
                printf("\nWARNING:  Solution appears to be infeasible!\n");
            return;
        }

        if(next_array[current_node]>0)
        {
            // Next node is somewhere in the middle of a route
            next_node_number = next_array[current_node];
            if(current_node!= dummy_index)
                num_in_route++;

            total_nodes++;
            current_node=next_node_number;
            cnt++;
        }
        else
        {
            // We must have a non-positive "next" node indicating the beginning of a new route
            i++;
            total_nodes++;
            num_in_route=0;

            route_start = - (next_array[current_node]);    
            current_route = route_num[route_start];
            current_node = route_start;

            printf("\nRoute %03d[0-%03d...%03d-0\tlen=%03.2f\tload=%04g\t#=%03d]",i,route[current_route].start,
                route[current_route].end,route[current_route].length,
                route[current_route].load,route[current_route].num_customers);
            cust_count+= route[current_route].num_customers;

            if(route[current_route].length>this->max_route_length || 
                route[current_route].load > this->max_veh_capacity)
                feasible=false;

            total_load += route[current_route].load;
            cnt++;
            if(current_node!= dummy_index)
                num_in_route++;
        }
    }
}

