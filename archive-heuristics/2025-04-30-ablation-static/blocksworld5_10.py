from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their correct
    position relative to their immediate support in the goal state, or that
    have the wrong block stacked directly on top of them. This captures
    discrepancies in the immediate block-on-block or block-on-table relationships
    required by the goal.

    # Assumptions
    - The goal state defines specific stacks or blocks on the table for all objects.
      Every block is either specified to be on another block or on the table.
    - The heuristic counts blocks that are 'misplaced' based on their immediate
      support and the block directly above them, compared to the goal configuration.
    - The cost of achieving the correct configuration is related to the number
      of blocks that are currently violating these immediate relationships.

    # Heuristic Initialization
    - Extracts the goal configuration to build maps representing the desired
      immediate support (`goal_pred`) and the desired block directly on top
      (`goal_succ`) for each block.
    - Collects the set of all blocks involved in the problem from the initial
      state and goal state facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to build maps representing the actual immediate
       support (`current_pred`) and the actual block directly on top
       (`current_succ`) for each block. Identify the block being held, if any.
       `current_pred[block]` will be the block it's on, 'table', or 'holding'.
       `current_succ[block]` will be the block on top, or 'clear'.
    2. Initialize the heuristic cost to 0.
    3. Iterate through each block identified in the problem:
       a. Determine the block's goal support (`goal_pred.get(block)`).
       b. Determine the block's goal successor (`goal_succ.get(block, 'clear')`).
       c. Determine the block's current support (`current_pred.get(block)`). If a block is not on another block, on the table, or held, its state is inconsistent with standard BW, but we handle it by checking if it's in `current_pred`.
       d. Determine the block's current successor (`current_succ.get(block, 'clear')`). Default to 'clear' if no block is on top.
       e. Check if the block is "misplaced" based on its support or the block above it:
          - If the block's current support (`current_pred.get(block)`) is different from its goal support (`goal_pred.get(block)`). Note: If a block is held, its current_pred is 'holding', which will likely differ from its goal_pred ('table' or another block), contributing to the cost.
          - OR if the block's current support is correct, but the block currently on top of it (`current_succ.get(block, 'clear')`) is different from the block that should be on top in the goal (`goal_succ.get(block, 'clear')`).
       f. If either of the conditions in step 3e is true, increment the heuristic cost.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configurations and all objects.
        """
        self.goals = task.goals

        self.goal_pred = {} # Maps block -> block/table it should be on
        self.goal_succ = {} # Maps block/table -> block that should be on it

        # Collect all objects from initial state and goal state facts
        self.all_objects = set()
        for fact in task.initial_state | task.goals:
             parts = get_parts(fact)
             # Add all parameters except 'table'
             for part in parts[1:]:
                 if part != 'table':
                     self.all_objects.add(part)

        # Parse goals to build goal_pred and goal_succ
        for goal in self.goals:
            if match(goal, "on", "*", "*"):
                b, a = get_parts(goal)[1:]
                self.goal_pred[b] = a
                self.goal_succ[a] = b
            elif match(goal, "on-table", "*"):
                b = get_parts(goal)[1]
                self.goal_pred[b] = 'table'
            # clear facts in goal indicate the top of a goal stack
            # We handle this after processing 'on' facts

        # For blocks that are not supposed to have anything on them in the goal
        # If a block is not a value in goal_succ, it should be clear in the goal.
        for obj in self.all_objects:
            if obj not in self.goal_succ:
                 self.goal_succ[obj] = 'clear'

        # The 'table' doesn't have a goal_succ in this formulation.
        # We only care about what is *on* a block.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        cost = 0

        current_pred = {} # Maps block -> block/table/holding it is on/in
        current_succ = {} # Maps block/table -> block currently on it
        holding_block = None

        # Parse current state to build current_pred and current_succ
        for fact in state:
            if match(fact, "on", "*", "*"):
                b, a = get_parts(fact)[1:]
                current_pred[b] = a
                current_succ[a] = b
            elif match(fact, "on-table", "*"):
                b = get_parts(fact)[1]
                current_pred[b] = 'table'
            elif match(fact, "holding", "*"):
                b = get_parts(fact)[1]
                current_pred[b] = 'holding'
                holding_block = b
            # We don't need to parse 'clear' or 'arm-empty' explicitly here,
            # as current_succ and holding_block capture the relevant state.

        # For blocks that are currently clear (nothing is on them)
        # Iterate through all objects that are not currently a key in current_succ
        # (meaning nothing is currently on them).
        for obj in self.all_objects:
             if obj not in current_succ:
                 current_succ[obj] = 'clear'


        # Calculate heuristic cost
        for block in self.all_objects:
            # Find goal position/support
            # Use .get() with a default in case a block is in all_objects but not in goal_pred
            # (e.g., if a block exists but isn't mentioned in goal on/on-table facts - unusual in BW)
            # Assuming all blocks are mentioned in goal on/on-table facts based on typical BW.
            goal_support = self.goal_pred.get(block)
            goal_top = self.goal_succ.get(block, 'clear') # Default to clear if not in goal_succ

            # Find current position/support
            current_support = current_pred.get(block) # None if block is not on/on-table/holding
            current_top = current_succ.get(block, 'clear') # Default to clear if not in current_succ

            # Check if block is on the wrong support
            is_wrong_support = False
            if goal_support is not None: # Block has a specified goal support
                if current_support != goal_support:
                     is_wrong_support = True
            # else: # Block's support is not specified in goal - consider any support ok unless holding?
            #    if current_support == 'holding': # Holding is likely wrong if goal support is unspecified
            #        is_wrong_support = True
            # Let's stick to the assumption that all blocks have goal support specified.

            # Check if block is on correct support but has wrong block on top
            is_wrong_top = False
            if not is_wrong_support: # Only check if support is correct
                 if current_top != goal_top:
                     is_wrong_top = True

            if is_wrong_support or is_wrong_top:
                cost += 1

        return cost
