#

ACCEPTANCE_MODES = [
    'ACCEPT',           # accept always
    'ACCEPT_EPSILON',   # accept if improvement > epsilon
    'SELECT',           # decide about acceptance
    'SELECT_EPSILON',   # decide only if improvement <= epsilon, else always accept
]

OPERATOR_MODES = [
    'SET',              # use specified set of one or more operators
    'ALL',              # use all available local search operators
    'SELECT_LS',        # select a specific operator
    'SELECT_LS+',       # select a specific operator (inc. perturbations)
]

POSITION_MODES = [
    'ALL',              # use all possible positions
    'RANDOM',           # use set of random positions
]
