from heuristics.heuristic_base import Heuristic

# Helper function to parse a fact string
def get_parts(fact_str):
    """Parses a PDDL fact string into predicate and arguments."""
    # Remove surrounding parentheses and split by space
    return fact_str[1:-1].split()

def parse_state(state_facts):
    """Parses a frozenset of state facts into structured data."""
    on_facts = {} # block -> block_it_is_on
    on_table_facts = set() # {block, ...}
    clear_facts = set() # {block, ...}
    holding_fact = None # block or None
    arm_empty_fact = False

    for fact_str in state_facts:
        parts = get_parts(fact_str)
        predicate = parts[0]
        args = parts[1:]

        if predicate == 'on':
            on_facts[args[0]] = args[1]
        elif predicate == 'on-table':
            on_table_facts.add(args[0])
        elif predicate == 'clear':
            clear_facts.add(args[0])
        elif predicate == 'holding':
            # Assuming only one block can be held at a time
            holding_fact = args[0]
        elif predicate == 'arm-empty':
            arm_empty_fact = True
        # Ignore other predicates if any, though blocksworld only has these

    return {
        'on': on_facts,
        'on-table': on_table_facts,
        'clear': clear_facts,
        'holding': holding_fact,
        'arm-empty': arm_empty_fact
    }

def parse_goal(goal_facts):
    """Parses a frozenset of goal facts into target configuration."""
    target_support = {} # block -> block_it_should_be_on or 'table'
    goal_clear = set() # {block, ...}

    for fact_str in goal_facts:
        parts = get_parts(fact_str)
        predicate = parts[0]
        args = parts[1:]

        if predicate == 'on':
            target_support[args[0]] = args[1]
        elif predicate == 'on-table':
            target_support[args[0]] = 'table'
        elif predicate == 'clear':
            goal_clear.add(args[0])
        # Ignore other goal predicates if any

    return target_support, goal_clear

def is_in_goal_stack_prefix(block, state_parsed, target_support, memo):
    """
    Checks if a block is currently in the correct position relative to the
    goal stack below it, recursively. Uses memoization.
    """
    if block in memo:
        return memo[block]

    # If the block is not part of any goal stack definition (i.e., not a key
    # in target_support), it doesn't contribute to the count of blocks *not*
    # in their goal stack prefix. We consider it "correct" in this context
    # as it doesn't need to be in a goal stack prefix.
    if block not in target_support:
        memo[block] = True
        return True

    target = target_support[block]

    # Find current support
    current_support = None
    if block in state_parsed['on']:
        current_support = state_parsed['on'][block]
    elif block in state_parsed['on-table']:
        current_support = 'table'
    # If holding, current_support is effectively 'arm', which is not a target support.
    # If under something, its support is still the block it's on, which is captured by state_parsed['on'].

    # Check if the immediate support is correct
    if current_support != target:
        memo[block] = False
        return False

    # If the immediate support is correct, recursively check the support below.
    # This handles the base case where target is 'table' (recursion stops).
    # Ensure the target block exists in target_support before recursive call
    # to prevent infinite recursion or errors on malformed goals (though PDDL goals are usually well-formed).
    # The check `if block not in target_support` at the start handles the base case
    # where the target is 'table' because 'table' is not a block and won't be a key in target_support.
    memo[block] = is_in_goal_stack_prefix(target, state_parsed, target_support, memo)
    return memo[block]


class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing two components:
        1. The number of blocks that are part of a goal stack or goal on-table condition
           but are not currently in the correct position relative to the goal stack
           prefix below them.
        2. The number of blocks that are required to be clear in the goal state
           but are not clear in the current state.

        The first component captures the structural correctness of the stacks,
        counting blocks that need to be moved to build the goal configuration
        from the bottom up. The second component captures the cost of clearing
        blocks that are obstructing goal conditions.

    Assumptions:
        - The input state and goal are valid for the Blocksworld domain.
        - The goal state defines acyclic stacks or blocks on the table.
        - The state representation is a frozenset of PDDL fact strings.
        - The heuristic is used for greedy best-first search and does not need
          to be admissible.

    Heuristic Initialization:
        The constructor parses the goal facts from the task object to precompute
        the target configuration. This includes identifying for each block
        whether it should be on another specific block or on the table
        (`target_support`), and which blocks are required to be clear
        in the goal state (`goal_clear`). This precomputation is done once
        when the heuristic is initialized. The Blocksworld domain has no static
        facts, so task.static is not used.

    Step-By-Step Thinking for Computing Heuristic:
        1. Parse the current state (a frozenset of fact strings) into a more
           structured representation (dictionaries and sets) for easier access
           to 'on', 'on-table', 'clear', 'holding', and 'arm-empty' facts.
        2. Initialize the heuristic value to 0.
        3. Compute the first component: Iterate through all blocks that are
           part of the target configuration (i.e., are keys in `self.target_support`).
           For each such block, check if it is currently in its "goal stack prefix".
           A block is in its goal stack prefix if its immediate support
           (the block it's on, or the table) matches its target support, AND
           the block it's supposed to be on (if any) is also in *its* goal
           stack prefix, recursively down to the table. This check uses a
           recursive helper function (`is_in_goal_stack_prefix`) with memoization
           to be efficient.
           If a block is *not* in its goal stack prefix, increment the heuristic value by 1.
        4. Compute the second component: Iterate through all blocks that are
           required to be clear in the goal state (`self.goal_clear`). For each such
           block, check if it is present in the 'clear' facts of the current state.
           If it is not clear, increment the heuristic value by 1.
        5. The total heuristic value is the sum of the counts from steps 3 and 4.
           A heuristic value of 0 is returned if and only if the state is a goal state.
    """
    def __init__(self, task):
        self.goals = task.goals
        # task.static is empty for blocksworld, no static info to extract
        self.target_support, self.goal_clear = parse_goal(self.goals)

    def __call__(self, node):
        state = node.state
        state_parsed = parse_state(state)

        h_value = 0

        # Component 1: Blocks not in their goal stack prefix
        memo = {} # Memoization for is_in_goal_stack_prefix
        for block in self.target_support:
            if not is_in_goal_stack_prefix(block, state_parsed, self.target_support, memo):
                h_value += 1

        # Component 2: Blocks that should be clear but are not
        for block in self.goal_clear:
            if block not in state_parsed['clear']:
                 h_value += 1

        return h_value
