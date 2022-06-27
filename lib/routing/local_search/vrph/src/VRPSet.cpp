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
//Added by MVG to set private data members (rather than read from file)

#include "VRPH.h"
//TODO: compare setting the data versus TSPLIB reader - to make sure
//  we are setting the data correctly.

void VRP::setMatrixSize(int matrixSize){
   matrix_size = matrixSize;
}

void VRP::setDummyIndex(int dummyIndex){
   dummy_index = dummyIndex;
}

void VRP::setVehicleCapacity(double capacity){
   max_veh_capacity = capacity;
   orig_max_veh_capacity = capacity;
}

void VRP::setDemand( const double * demand){
   if(!demand)
      return;
   //The demand sent in assumes the depot=0 and customers=1..n
   int n = num_nodes;
   for(int i = 0; i <= n; i++){
      nodes[i].id     = i+1;
      nodes[i].demand = demand[i];
   }
   assert(dummy_index == n+1);
   nodes[n+1].demand=0;//dummyIndex(=n+1)      
}

void VRP::setDistanceMatrix(const double * distDense,
                            int            isUnDirected){
   //TODO: OOM check
   if(!distDense || d)
      return;
   //The distDense sent assumes the depot=0 and
   //   we have a complete undirected graph.
   int i,j;
   int n      = num_nodes;//this is really number of customers
   int nNodes = n+1;      //== dummy_index
   int temp = nNodes + 1;

   //TODO: OOM check
   d = new double* [temp];
   d[0] = new double [(temp)*(temp)];
   for(i = 1; i < nNodes+1; i++)
      d[i] = d[i-1] + (temp);

   if(isUnDirected){
      int index = 0;
      for(i = 1; i < nNodes; i++){
         for(j = 0; j < i; j++){
            //NOTE: this assumes symmetric
            d[i][j] = distDense[index];
            d[j][i] = distDense[index];
            index++;
         }
      }
   }
   else{
      symmetric=false;
      int index = 0;
      for(i = 0; i < nNodes; i++){
         for(j = 0; j < nNodes; j++){
            //NOTE: this assumes symmetric
            d[i][j] = distDense[index];
            index++;
         }
      }
   }

   //set self-links to 0
   for(i = 0; i <= nNodes; i++)
      d[i][i] = 0;

   
   //The dummy_index is at the depot.
   for(i = 0; i < nNodes; i++){
      d[dummy_index][i] = d[0][i];
      d[i][dummy_index] = d[i][0];
   }   
}

void VRP::setBreakFunction(int (*checkBreakFuncU)(void*),
                           void * checkBreakFuncDataU){
   checkBreakFunc = checkBreakFuncU;
   checkBreakFuncData = checkBreakFuncDataU;   
}
