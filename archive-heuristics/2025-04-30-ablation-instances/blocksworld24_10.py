from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions from Logistics example
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(on a b)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if lengths match, unless pattern ends with '*'
    if len(parts) != len(args) and (len(args) == 0 or args[-1] != '*'):
         return False
    # Check if parts match args (up to the length of args)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed by summing three components:
    1. The number of blocks that are not part of a correctly built stack from the bottom up,
       relative to their desired position in the goal stacks.
    2. The number of blocks that are currently on top of another block, where this 'on'
       relationship is not desired in the goal.
    3. A penalty if the robot's arm is not empty.

    # Assumptions
    - The goal state defines the desired configuration of blocks on top of each other or the table.
    - Blocks not mentioned as being 'on' another block or 'on-table' in the goal are
      considered irrelevant to the goal stack structure for the first heuristic component (h1).
    - Standard Blocksworld goals require the arm to be empty in the goal state.

    # Heuristic Initialization
    - Extracts the desired support (another block or 'table') for each block involved
      as the upper block in goal 'on' facts or blocks that are 'on-table' in the goal.
      This information is stored in `self.goal_pos`.
    - Stores the set of desired 'on' facts from the goal as strings for quick lookup
      in the second heuristic component (h2).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Extract Relevant State Information:
       - Identify the current support (block or 'table') for every block currently
         on another block or on the table.
       - Determine if the robot's arm is empty or holding a block.
       - Collect all current 'on' facts as strings.

    2. Compute the First Component (h1 - Misplaced in Goal Stacks):
       - This component counts blocks whose position deviates from the goal stack structure.
       - Define a recursive helper function `is_correctly_placed(block)`:
         - Base case: `is_correctly_placed('table')` is always True.
         - If the block's position is not specified in the goal stacks (i.e., it's not
           a key in `self.goal_pos`), its position doesn't break a goal stack requirement,
           so consider it True in this context.
         - If the block is in `self.goal_pos`:
           - Find its `goal_support` from `self.goal_pos`.
           - Find its `current_support` from the state.
           - The block is correctly placed if and only if its `current_support` matches
             its `goal_support` AND its `goal_support` is also correctly placed
             (recursive call).
       - Use memoization within `is_correctly_placed` for efficiency.
       - Iterate through all blocks that are keys in `self.goal_pos`. For each such block,
         if `is_correctly_placed(block)` returns False, increment h1.

    3. Compute the Second Component (h2 - Wrong 'On' Relations):
       - This component counts 'on' relationships in the current state that are not
         desired in the goal state. Each such relationship implies a block is on
         another block when it shouldn't be, requiring at least one unstack/move action.
       - Iterate through all facts in the current state. If a fact is an 'on' predicate,
         check if this exact 'on' fact string exists in the set of goal 'on' facts.
         If it does not, increment h2.

    4. Compute the Third Component (h3 - Arm Penalty):
       - This component penalizes a non-empty arm, encouraging the agent to put down
         whatever it is holding to free the arm for necessary moves.
       - If the state contains `(arm-empty)`, h3 is 0. Otherwise (if `(holding ?x)` is present), h3 is 1.
       - This assumes standard Blocksworld goals where `(arm-empty)` is desired.

    5. Calculate Total Heuristic:
       - The total heuristic value for the state is the sum `h1 + h2 + h3`.
       - This value is 0 if and only if the state is a standard goal state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals

        # Extract goal positions for blocks involved in goal stacks
        # goal_pos[block] = support_block or 'table'
        # This maps a block (that should be on top) to what it should be on.
        self.goal_pos = {}
        # Store goal 'on' facts as strings for quick lookup in h2
        self.goal_on_facts_str = set()

        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] == 'on':
                block, support = parts[1], parts[2]
                self.goal_pos[block] = support
                self.goal_on_facts_str.add(goal_fact)
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_pos[block] = 'table'
            # Ignore other goal predicates like (clear ?) or (arm-empty) for goal_pos

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Extract current state information ---
        current_on = {} # {block: support} for (on block support)
        current_on_table = set() # {block} for (on-table block)
        arm_is_empty = False
        # holding_block = None # Not strictly needed, just check arm_empty

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                current_on[parts[1]] = parts[2]
            elif parts[0] == 'on-table':
                current_on_table.add(parts[1])
            elif parts[0] == 'arm-empty':
                arm_is_empty = True
            # elif parts[0] == 'holding':
            #     holding_block = parts[1]

        # --- Compute h1: Blocks not in correctly built stacks ---
        # Memoization for the recursive check
        memo_correctly_placed = {}

        def is_correctly_placed(block):
            """
            Recursively checks if a block is in its goal position AND
            the block below it is also correctly placed, down to the table.
            Considers blocks not in self.goal_pos as valid bases for goal stacks.
            """
            if block == 'table':
                return True # Base case: the table is always correctly placed

            if block in memo_correctly_placed:
                return memo_correctly_placed[block]

            # If the block's position is not specified in the goal stacks (i.e.,
            # it's not a key in self.goal_pos), it doesn't have a required position
            # relative to the goal structure. Treat it as a valid base for the
            # stack above it.
            if block not in self.goal_pos:
                 memo_correctly_placed[block] = True
                 return True

            # Block is in self.goal_pos, so its position is specified in the goal.
            goal_support = self.goal_pos[block]

            # Find current support
            current_support = None
            if block in current_on_table:
                current_support = 'table'
            elif block in current_on:
                current_support = current_on[block]
            # If block is held, current_support is None

            # Check if the block is on the correct support AND the support is correctly placed
            # If current_support is None (block is held), it's not correctly placed relative to a support.
            is_correct = (current_support == goal_support) and is_correctly_placed(goal_support)

            memo_correctly_placed[block] = is_correct
            return is_correct

        h1 = 0
        # Iterate over all blocks that are the 'top' block in a goal 'on' fact
        # or a block that is 'on-table' in the goal. These are the keys in goal_pos.
        for block in self.goal_pos.keys():
             if not is_correctly_placed(block):
                 h1 += 1

        # --- Compute h2: Blocks wrongly placed on top of others ---
        h2 = 0
        # Iterate through all 'on' facts in the current state
        for fact in state:
            if match(fact, "on", "*", "*"):
                # Check if this 'on' fact is NOT in the goal 'on' facts
                if fact not in self.goal_on_facts_str:
                    h2 += 1 # Count this misplaced 'on' relationship

        # --- Compute h3: Penalty for arm not empty ---
        h3 = 0
        if not arm_is_empty:
             h3 = 1

        # Total heuristic is the sum
        total_cost = h1 + h2 + h3

        return total_cost
