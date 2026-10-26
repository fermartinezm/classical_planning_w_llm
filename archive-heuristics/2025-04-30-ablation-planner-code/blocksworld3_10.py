from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty fact string case defensively
    if not fact or len(fact) < 2:
        return []
    return fact[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by summing costs for unsatisfied goal conditions. It penalizes for blocks
    that are not in their goal position/state and for blocks that obstruct
    the placement of other blocks.

    # Assumptions
    - The base cost of achieving a desired 'on' or 'on-table' fact is estimated as 2 actions (pickup/unstack + putdown/stack).
    - The base cost of achieving a desired 'clear' fact is estimated as 2 actions (unstacking the block on top and putting it down/stacking it elsewhere).
    - The heuristic sums these costs for each unsatisfied goal fact. If a block obstructs multiple unsatisfied goals, its clearing cost is counted multiple times, making the heuristic non-admissible but potentially better for greedy search by strongly penalizing states with many obstructions or unsatisfied goals.

    # Heuristic Initialization
    - Stores the set of goal facts for quick lookup.
    - Identifies all objects (blocks) present in the initial state or goal state to distinguish blocks from the 'table' concept.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is calculated as the sum of costs for each unsatisfied goal fact:

    1. Initialize heuristic value `h = 0`.
    2. Parse the current state facts:
       - Create a set of `(on X Y)` facts for quick lookup.
       - Create a set of `(on-table Z)` facts for quick lookup.
       - Create a set `not_clear_blocks` containing all blocks `underob` for which `(on ob underob)` is true in the current state. A block is clear if it is NOT in this set.
    3. Iterate through each goal fact `g` in the task's goals:
       - If `g` is already present in the current state, continue to the next goal fact (cost is 0 for this goal).
       - If `g` is `(on X Y)` and is not in the current state:
         - Add 2 to `h` (base cost to place X on Y).
         - If `X` is not clear in the current state (check `X in not_clear_blocks`):
           - Add 2 to `h` (cost to clear X).
         - If `Y` is a block (check `Y in self.objects`) and `Y` is not clear in the current state (check `Y in not_clear_blocks`):
           - Add 2 to `h` (cost to clear Y).
       - If `g` is `(on-table Z)` and is not in the current state:
         - Add 2 to `h` (base cost to put Z on the table).
         - If `Z` is not clear in the current state (check `Z in not_clear_blocks`):
           - Add 2 to `h` (cost to clear Z).
       - If `g` is `(clear W)` and is not in the current state:
         - If `W` is not clear in the current state (check `W in not_clear_blocks`):
           - Add 2 to `h` (cost to clear W by moving the block on top).
    4. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and identifying objects.
        """
        # Store goal facts as a set for quick lookup
        self.goals = set(task.goals)

        # Collect all objects (blocks) mentioned in init or goals
        self.objects = set()
        # Iterate through all facts in init and goals
        for fact_str in task.init + list(task.goals):
             parts = get_parts(fact_str)
             if parts:
                 predicate = parts[0]
                 # Objects appear as arguments in these predicates
                 if predicate in ['on', 'on-table', 'clear', 'holding']:
                     # Arguments are objects, except 'table' which is implicit
                     for arg in parts[1:]:
                         if arg != 'table': # 'table' is not an object name
                             self.objects.add(arg)

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state from the current state.
        """
        state = node.state  # Current world state (frozenset of fact strings)

        # Parse current state facts for efficient lookup
        current_on = set()
        current_on_table = set()

        # Build a set of blocks that are NOT clear
        not_clear_blocks = set()
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                # ob, underob = parts[1], parts[2]
                current_on.add(fact)
                underob = parts[2]
                not_clear_blocks.add(underob) # The block underneath is not clear
            elif predicate == 'on-table' and len(parts) == 2:
                current_on_table.add(fact)
            # 'clear', 'holding', 'arm-empty' facts are not needed for direct lookup
            # in this heuristic's calculation logic, as 'not_clear_blocks' covers
            # the 'clear' status derived from 'on' facts.

        # Helper function to check if a block is clear in the current state
        def is_clear(block):
            # A block is clear if nothing is on top of it according to current_on facts
            # This is equivalent to checking if the block is NOT in the set of blocks
            # that have something on top of them.
            return block not in not_clear_blocks

        heuristic_value = 0

        # Iterate through goal facts and add cost for unsatisfied ones
        for goal_fact in self.goals:
            # Check if the goal fact is already satisfied in the current state
            if goal_fact in state:
                continue # Goal fact is already satisfied, cost is 0 for this goal

            # If the goal fact is not satisfied, calculate its contribution to the heuristic
            parts = get_parts(goal_fact)
            if not parts: continue # Skip empty or malformed goal facts

            predicate = parts[0]

            if predicate == 'on' and len(parts) == 3:
                x, y = parts[1], parts[2]
                # Base cost to achieve (on X Y) is 2 actions (pickup/unstack X + stack X Y)
                heuristic_value += 2

                # Add cost to clear X if it's not clear
                if not is_clear(x):
                    heuristic_value += 2

                # Add cost to clear Y if it's a block and not clear
                # Y is a block if it's in our set of objects (i.e., not 'table')
                if y in self.objects and not is_clear(y):
                     heuristic_value += 2

            elif predicate == 'on-table' and len(parts) == 2:
                z = parts[1]
                # Base cost to achieve (on-table Z) is 2 actions (pickup/unstack Z + putdown Z)
                heuristic_value += 2

                # Add cost to clear Z if it's not clear
                if not is_clear(z):
                    heuristic_value += 2

            elif predicate == 'clear' and len(parts) == 2:
                w = parts[1]
                 # Cost to achieve (clear W) if it's not clear is 2 actions
                 # (unstack block_on_top_of_W W + putdown/stack block_on_top_of_W)
                if not is_clear(w):
                    heuristic_value += 2

            # Note: We typically ignore 'arm-empty' and 'holding' goals if they exist,
            # as they represent transient states during plan execution and are not
            # structural goals about block arrangement. This heuristic focuses on
            # the block configuration goals ('on', 'on-table', 'clear').

        return heuristic_value
