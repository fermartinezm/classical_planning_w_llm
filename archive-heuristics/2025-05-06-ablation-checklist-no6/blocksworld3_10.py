from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Recursive helper function to calculate the cost to clear a block
def cost_to_clear_recursive(b, current_block_on_top, clear_in_state, memo):
    """
    Calculates the estimated cost (number of actions) to make block 'b' clear.
    Assumes each block on top needs 2 actions (unstack/pickup + putdown) to move.
    Uses memoization to avoid redundant calculations.
    """
    if b in memo:
        return memo[b]

    if b in clear_in_state:
        memo[b] = 0
        return 0

    # Find the block t directly on top of b
    t = None
    for top_b, under_b in current_block_on_top.items():
        if under_b == b:
            t = top_b
            break

    # In a valid blocksworld state, if a block is not clear, there is exactly one block on top.
    # If t is None here, it indicates an issue with state parsing or an invalid state.
    # Assuming valid states, t will be found.
    # Cost to clear b = cost to clear t + cost to move t (estimated as 2 actions).
    cost = cost_to_clear_recursive(t, current_block_on_top, clear_in_state, memo) + 2
    memo[b] = cost
    return cost


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    Estimates the number of actions needed to reach the goal state.
    The heuristic is the sum of:
    1. 2 actions for each block that is not on its correct goal support.
    2. The cost to clear blocks that are not clear but need to be moved (because they are misplaced)
       or need to be clear as a base for an unsatisfied goal stack, or need to be clear according to the goal.
       The cost to clear a block is estimated recursively based on the tower above it.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and identifying all blocks.
        """
        self.goals = task.goals

        # Parse goal facts to build goal structure
        self.goal_support = {} # block -> block_below or 'table'
        self.goal_block_on_top = {} # block_below -> block_on_top (only for blocks, not 'table')
        self.goal_clear = set() # blocks that should be clear in the goal

        # Collect all blocks mentioned in the goals and initial state
        all_blocks_set = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                b_on_top, b_below = parts[1], parts[2]
                self.goal_support[b_on_top] = b_below
                self.goal_block_on_top[b_below] = b_on_top
                all_blocks_set.add(b_on_top)
                all_blocks_set.add(b_below)
            elif parts[0] == 'on-table':
                b = parts[1]
                self.goal_support[b] = 'table'
                all_blocks_set.add(b)
            elif parts[0] == 'clear':
                b = parts[1]
                self.goal_clear.add(b)
            # Ignore 'arm-empty' goals for this heuristic structure

        # Add blocks from initial state
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] in ['on', 'on-table', 'clear', 'holding']:
                 for obj in parts[1:]:
                     all_blocks_set.add(obj)
             elif parts[0] == 'arm-empty':
                 pass # Ignore arm-empty fact itself

        self.all_blocks = list(all_blocks_set) # Store as list if order matters, set is fine too.

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Parse current state facts
        current_support = {} # block -> block_below or 'table' or 'arm'
        current_block_on_top = {} # block_below -> block_on_top (only for blocks)
        clear_in_state = set() # set of clear blocks
        holding_in_state = None # the block being held, or None
        # arm_empty_in_state = False # Not directly used in H calculation

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                b_on_top, b_below = parts[1], parts[2]
                current_support[b_on_top] = b_below
                current_block_on_top[b_below] = b_on_top
            elif parts[0] == 'on-table':
                b = parts[1]
                current_support[b] = 'table'
            elif parts[0] == 'clear':
                b = parts[1]
                clear_in_state.add(b)
            elif parts[0] == 'holding':
                b = parts[1]
                holding_in_state = b
            # elif parts[0] == 'arm-empty':
            #     arm_empty_in_state = True # Not directly used

        H = 0
        memo = {} # Memoization dictionary for cost_to_clear_recursive

        # Identify all blocks present in the current state
        current_blocks_present = set(current_support.keys())
        if holding_in_state:
            current_blocks_present.add(holding_in_state)
        # Consider all blocks from goal and current state
        all_relevant_blocks = set(self.all_blocks) | current_blocks_present


        # Set of blocks that are not on their goal support
        misplaced_support = set()
        for b in all_relevant_blocks:
            goal_supp = self.goal_support.get(b) # None if b is not a key in goal_support
            curr_supp = current_support.get(b)

            # A block is misplaced if it's a key in goal_support and its current support doesn't match.
            # OR if it's present but not a key in goal_support (meaning it should be on the table and clear)
            # and it's not currently on the table.
            if b in self.goal_support:
                if curr_supp != goal_supp:
                    misplaced_support.add(b)
            elif b in current_blocks_present: # Block is in state but not a key in goal_support
                 # Implicit goal is on-table and clear. Check if it's on the table.
                 if curr_supp != 'table':
                      misplaced_support.add(b)


        # Set of blocks that are bases for unsatisfied (on x y) goals and are not clear
        bases_to_clear = set()
        for x, y in self.goal_support.items():
            # If (on x y) is a goal
            # Check if the goal fact (on x y) is NOT in the current state.
            goal_fact_on_xy = f'(on {x} {y})'
            if goal_fact_on_xy not in state:
                 # This goal is not satisfied. The base 'y' needs to be clear to stack 'x' on it.
                 if y != 'table' and y in all_relevant_blocks and y not in clear_in_state:
                     bases_to_clear.add(y)

        # Cost for each block that is not on its goal support
        for b in misplaced_support:
            # Cost to move b = 1 (pickup/unstack) + 1 (stack/putdown) = 2
            # Cost to clear b = cost_to_clear_recursive(b, ...)
            # If b is held, cost_to_clear_recursive(b) is 0. Cost is 2.
            # If b is on something, cost_to_clear_recursive(b) is calculated.
            H += 2 + cost_to_clear_recursive(b, current_block_on_top, clear_in_state, memo)

        # Cost for each base that needs clearing for an unsatisfied goal, if not already counted
        for b in bases_to_clear:
            if b not in misplaced_support: # Avoid double counting if b is also misplaced
                 H += cost_to_clear_recursive(b, current_block_on_top, clear_in_state, memo)

        # Cost for blocks that should be clear in the goal but are not clear in the state.
        for b in self.goal_clear:
            if b in all_relevant_blocks and b not in clear_in_state:
                # Block b should be clear but isn't. It needs clearing.
                # Add cost to clear b, but avoid double counting if b is already handled.
                if b not in misplaced_support and b not in bases_to_clear:
                     H += cost_to_clear_recursive(b, current_block_on_top, clear_in_state, memo)

        # Note: The arm-empty goal is implicitly handled. If (holding b) is true,
        # b is added to misplaced_support, adding 2 + cost_to_clear(b) to H.
        # Since holding, cost_to_clear(b)=0. Cost is 2. This covers the cost
        # of putting the block down and potentially picking it up again later.

        return H
