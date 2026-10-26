# This is the domain-dependent heuristic for the blocksworld domain.
# It should be placed in a file named blocksworldHeuristic.py
# and assumes the 'heuristics.heuristic_base' module is available.

# from fnmatch import fnmatch # Not strictly needed for this implementation
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty string or malformed fact gracefully
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are either not in their
    correct position within the goal stack structure or are preventing a block
    from being clear when it needs to be clear in the goal state. It is designed
    to guide a greedy best-first search and is not admissible.

    # Assumptions:
    - The goal state defines a specific configuration of blocks, typically
      forming stacks on the table, and potentially requires certain blocks
      to be clear.
    - All blocks involved in the problem are present in the initial state.
    - The heuristic counts two types of "misplacedness" or "goal violations":
        1. A block is not in its correct recursive goal position (i.e., it's
           not on the correct block or table, or the block it's on is not
           correctly placed below it according to the goal structure).
        2. A block that is required to be clear in the goal state is currently
           not clear.

    # Heuristic Initialization
    - Parses the goal state to determine the desired support for each block
      (which block it should be on, or if it should be on the table) by
      examining `(on ?x ?y)` and `(on-table ?x)` goal predicates.
    - Identifies which blocks need to be clear in the goal state by examining
      `(clear ?x)` goal predicates.
    - Collects all blocks present in the initial state to know the full set
      of objects in the problem.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Initialize the total heuristic cost to 0.
    2. Parse the current state to build data structures representing the
       current configuration:
       - `current_support`: A mapping from each block to what it is currently
         on (another block, 'table', or 'holding').
       - `current_clear`: A set of blocks that are currently clear.
    3. Part 1: Count blocks not in their correct goal stack position.
       - Define a recursive helper function `is_correctly_placed(block)`
         that checks if a block is in its final goal position relative to
         its support, and if its support (if it's another block) is also
         correctly placed, recursively down to the table.
         - If a block is not explicitly part of the goal support structure
           (i.e., not in `self.goal_support`), it is considered correctly
           placed relative to support if it is not currently being held.
         - Use memoization within the recursive function to avoid redundant
           computations for blocks in the same stack.
       - Iterate through all blocks in the problem. If a block is not
         `is_correctly_placed`, increment the total cost.
    4. Part 2: Count blocks blocking goal clear conditions.
       - Iterate through the set of blocks that must be clear in the goal
         (`self.goal_clear_blocks`).
       - If a block from this set is not present in the `current_clear` set
         (meaning something is on top of it), increment the total cost.
    5. After computing the `total_cost` based on the two parts, check if the
       current state is the actual goal state (by checking if `self.goals`
       is a subset of the state facts). If it is the goal state, the heuristic
       value must be 0. Otherwise, return the calculated `total_cost`. This
       ensures the heuristic is 0 if and only if the goal is reached.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and identifying all blocks.
        """
        self.goals = task.goals
        self.initial_state = task.initial_state

        # Map blocks to their goal support (block or 'table')
        self.goal_support = {}
        # Set of blocks that must be clear in the goal
        self.goal_clear_blocks = set()
        # Set of all blocks in the problem
        self.all_blocks = set()

        # Parse goals to build goal_support and goal_clear_blocks, and find blocks
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, support = parts[1], parts[2]
                self.goal_support[block] = support
                self.all_blocks.add(block)
                self.all_blocks.add(support)
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                self.goal_support[block] = 'table'
                self.all_blocks.add(block)
            elif predicate == "clear" and len(parts) == 2:
                block = parts[1]
                self.goal_clear_blocks.add(block)
                self.all_blocks.add(block)
            # Ignore other goal predicates if any

        # Parse initial state to find any blocks not mentioned in goals
        # (Robustness: ensures all blocks in the problem are considered)
        for fact in self.initial_state:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]
             if predicate in ["clear", "on-table", "holding"] and len(parts) == 2:
                 self.all_blocks.add(parts[1])
             elif predicate == "on" and len(parts) == 3:
                 self.all_blocks.add(parts[1])
                 self.all_blocks.add(parts[2])
             # Ignore 'arm-empty' and other potential predicates

    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # Build current state representation
        current_support = {} # block -> support (block, 'table', or 'holding')
        current_clear = set() # set of clear blocks
        # current_on_top is not strictly needed with the recursive check,
        # but we do need current_clear for Part 2.

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, support = parts[1], parts[2]
                current_support[block] = support
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                current_support[block] = 'table'
            elif predicate == "holding" and len(parts) == 2:
                block = parts[1]
                current_support[block] = 'holding'
            elif predicate == "clear" and len(parts) == 2:
                 current_clear.add(parts[1])

        total_cost = 0
        memo_correctly_placed = {} # Memoization for recursive calls

        # Helper function to check if a block is in its correct goal stack position
        def is_correctly_placed(block):
            if block in memo_correctly_placed:
                return memo_correctly_placed[block]

            # If block is not in goal_support, it doesn't have a specific goal stack position.
            # Consider it "correctly placed" relative to support if it's not held.
            if block not in self.goal_support:
                 current_sup = current_support.get(block)
                 result = (current_sup != 'holding')
                 memo_correctly_placed[block] = result
                 return result

            # Block is in goal_support, check its position relative to its goal support
            goal_sup = self.goal_support[block]
            current_sup = current_support.get(block)

            # If block is not in the state in a standard position (shouldn't happen) or is held
            if current_sup is None or current_sup == 'holding':
                result = False
            # If block is on the table
            elif current_sup == 'table':
                result = (goal_sup == 'table')
            # If block is on another block
            elif current_sup in self.all_blocks: # Check if current_sup is a block
                 result = (goal_sup == current_sup) and is_correctly_placed(current_sup)
            else: # Should not happen in blocksworld with valid states
                 result = False

            memo_correctly_placed[block] = result
            return result

        # Part 1: Count blocks not in their correct goal stack position.
        for block in self.all_blocks:
             if not is_correctly_placed(block):
                 total_cost += 1

        # Part 2: Count blocks blocking goal clear conditions.
        for block_to_be_clear in self.goal_clear_blocks:
            if block_to_be_clear not in current_clear:
                # This block needs to be clear but isn't. Add cost.
                total_cost += 1

        # Heuristic must be 0 in the goal state.
        # If the state is the goal state, return 0. Otherwise, return the calculated cost.
        # This check ensures h=0 iff state is goal.
        if self.goals <= state:
             return 0

        return total_cost
