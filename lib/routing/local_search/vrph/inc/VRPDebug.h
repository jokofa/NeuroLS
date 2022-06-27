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

#ifndef _VRP_DEBUG_H
#define _VRP_DEBUG_H

#define DEBUG_ALL               0
#define VERIFY_ALL              0

#define TSPLIB_DEBUG            0 + DEBUG_ALL
#define SOLVER_DEBUG            0 + DEBUG_ALL

#define FIXED_DEBUG             0 + DEBUG_ALL
#define SEARCH_DEBUG            0 + DEBUG_ALL
#define VRPH_TABU_DEBUG         0 + DEBUG_ALL
#define BLOAT_DEBUG             0 + DEBUG_ALL
#define WAREHOUSE_DEBUG         0 + DEBUG_ALL

#define CW_DEBUG                0 + DEBUG_ALL
#define CW_VERIFY               0 + VERIFY_ALL
#define CLEAN_DEBUG             0 + DEBUG_ALL

#define SWEEP_DEBUG             0 + DEBUG_ALL
#define SWEEP_VERIFY            0 + VERIFY_ALL

#define Q_DEBUG                 0 + DEBUG_ALL
#define Q_VERIFY                0 + VERIFY_ALL

#define OP_VERIFY               0 + DEBUG_ALL

#define STRING_DEBUG            0 + DEBUG_ALL
#define STRING_VERIFY           0 + VERIFY_ALL

#define OPM_VERIFY              0 + VERIFY_ALL
#define OPM_DEBUG               0 + DEBUG_ALL

#define OR_VERIFY               0 + VERIFY_ALL
#define OR_DEBUG                0 + DEBUG_ALL

#define POSTSERT_VERIFY         1 + VERIFY_ALL
#define POSTSERT_DEBUG          1 + DEBUG_ALL

#define PRESERT_VERIFY          1 + VERIFY_ALL
#define PRESERT_DEBUG           1 + DEBUG_ALL

#define FLIP_DEBUG              0 + DEBUG_ALL
#define FLIP_VERIFY             0 + VERIFY_ALL

#define SWAP_ENDS_DEBUG         0 + DEBUG_ALL
#define SWAP_ENDS_VERIFY        0 + VERIFY_ALL

#define SWAP_DEBUG              0 + DEBUG_ALL
#define SWAP_VERIFY             0 + VERIFY_ALL

#define REVERSE_DEBUG           0 + DEBUG_ALL
#define REVERSE_VERIFY          0 + VERIFY_ALL

#define SWAP_VERIFY             0 + VERIFY_ALL
#define SWAP_DEBUG              0 + DEBUG_ALL

#define CONCATENATE_DEBUG       0 + DEBUG_ALL
#define CONCATENATE_VERIFY      0 + VERIFY_ALL


#define TPM_DEBUG               0 + DEBUG_ALL
#define TPM_VERIFY              0 + VERIFY_ALL

#define TWO_OPT_DEBUG           0 + DEBUG_ALL
#define TWO_OPT_VERIFY          0 + VERIFY_ALL

#define THREE_OPT_DEBUG         0 + DEBUG_ALL
#define THREE_OPT_VERIFY        0 + VERIFY_ALL

#define CROSS_EXCHANGE_DEBUG    0 + DEBUG_ALL
#define CROSS_EXCHANGE_VERIFY   0 + VERIFY_ALL


#define NEIGHBOR_DEBUG          0 + DEBUG_ALL

#define EJECT_DEBUG             0 + DEBUG_ALL


void report_error(const char* format, ...);
// Just sends message to stderr and exits.



#endif



