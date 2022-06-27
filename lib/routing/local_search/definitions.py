
# Various rules that can be or'ed together

# Accept only those moves that decrease the total route length.
VRPH_DOWNHILL = 1
# Accept moves according to the record-to-record acceptance strategy
VRPH_RECORD_TO_RECORD = 1<<1
# Accept moves according to the simulated annealing metaheuristic
VRPH_SIMULATED_ANNEALING = 1<<2

# Make the first solution modification that is found
# that meets the other specifications given in the current rules.
VRPH_FIRST_ACCEPT = 1<<3
# Evaluate all moves in the neighborhood and make
# the move that leads to a feasible solution with
# minimum total route length (this is the best move).
VRPH_BEST_ACCEPT = 1<<4
# If any improving move is found, make this move.
# If no improving move is found, then make the
# move that increases the total route length by the smallest amount.
VRPH_LI_ACCEPT = 1<<5

# Search for moves that involve the modification of more than one route.
VRPH_INTER_ROUTE_ONLY = 1<<6
# Search for moves that involve only a single route.
VRPH_INTRA_ROUTE_ONLY = 1<<7
# Limit the search to those moves that only involve
# a particular node’s neighbor list.
VRPH_USE_NEIGHBOR_LIST = 1<<8

# ...
VRPH_FREE = 1<<9
# ...
VRPH_BALANCED = 1<<10

# Create an ordered solution buffer by concatenating all routes
# and search for moves that involve
# nodes that are found when moving forward in the solution buffer
VRPH_FORWARD = 1<<11
# Create an ordered solution buffer by concatenating all the routes
# and search for moves that involve
# nodes that are found when moving backward in the solution buffer
VRPH_BACKWARD = 1<<12
# When examining the search neighborhood for
# moves, evaluate these moves in a random order.
VRPH_RANDOMIZED = 1<<13

# When comparing two moves, evaluate them by
# considering only the savings or improvement offered by the moves.
VRPH_SAVINGS_ONLY = 1<<14
# When comparing two moves that involve more than a single route,
# attempt to minimize the number of routes by trying to maximize the sum of the
# squares of the number of nodes in the routes involved in the move.
VRPH_MINIMIZE_NUM_ROUTES = 1<<15

# Forbid those moves that disrupt any edges that are currently set as fixed.
VRPH_FIXED_EDGES = 1<<17
# ...
VRPH_ALLOW_INFEASIBLE = 1<<18
# ...
VRPH_NO_NEW_ROUTE = 1<<19
# Forbid those moves that result in routes that are
# tabu according to the solution’s current memory.
VRPH_TABU = 1<<20


# Heuristic operations
ONE_POINT_MOVE = 1<<21
TWO_POINT_MOVE = 1<<22
TWO_OPT = 1<<23
OR_OPT = 1<<24
THREE_OPT = 1<<25
CROSS_EXCHANGE = 1<<26
THREE_POINT_MOVE = 1<<27

# custom rules
# Perturbs the current solution as suggested in "Li et al."
PERTURB_LI = 1<<28
#  Perturbs the existing solution by attempting to move num different random
#  nodes into new positions using Osman parameter alpha.
#  Gives up after attempting 2*V.num_nodes moves.
PERTURB_OSMAN = 1<<29
# Perturbs the existing solution by attempting to
# move num nodes from the provided nodelist (required!)
# into new positions using Osman parameter alpha.
PERTURB_TARGET = 1<<30
# -> 1<<31 is not in signed int32 precision anymore!

# Values of heuristics that can be used in functions such as 
# clean_routes

NUM_HEURISTICS = 7
ONE_POINT_MOVE_INDEX = 0  
TWO_POINT_MOVE_INDEX = 1
TWO_OPT_INDEX = 2
OR_OPT_INDEX = 3
THREE_OPT_INDEX = 4
CROSS_EXCHANGE_INDEX = 5
THREE_POINT_MOVE_INDEX = 6


# Move types
PRESERT = 1
POSTSERT = 2
CONCATENATE = 3
SWAP_ENDS = 4
FLIP = 5
MOVE_STRING = 6
SWAP = 7


# Supported TSPLIB Problem types
VRPH_TSP = 1
VRPH_CVRP = 2

# Supported TSPLIB Edge Weight Formats
VRPH_FUNCTION = 1
VRPH_UPPER_ROW = 2
VRPH_FULL_MATRIX = 3
VRPH_LOWER_ROW = 4
VRPH_UPPER_DIAG_ROW = 5
VRPH_LOWER_DIAG_ROW = 6

# Supported TSPLIB Coord types
VRPH_TWOD_COORDS = 2
VRPH_THREED_COORDS = 3

# Supported TSPLIB Edge Weight Types
VRPH_EXPLICIT = 0
VRPH_EUC_2D = 1
VRPH_EUC_3D = 2
VRPH_MAX_2D = 3
VRPH_MAX_3D = 4
VRPH_MAN_2D = 5
VRPH_MAN_3D = 6
VRPH_CEIL_2D = 7
VRPH_GEO = 8
VRPH_EXACT_2D = 9

